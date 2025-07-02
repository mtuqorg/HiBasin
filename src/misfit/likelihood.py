import numpy as np
from numpy.fft import fft, ifft, fftfreq
from util.math import exponential_covariance, calc_InversionDeterminant_cd

ns,nc,nt = 8,3,150
cov_matrix = exponential_covariance(150, 4)
cov_d = np.broadcast_to(cov_matrix, (ns, nc, nt, nt))
cov_inv, log_cov_det = calc_InversionDeterminant_cd(cov_d)

class Loglikelihood:
    def __init__(self, data, method='full_mij_uncorrelated'):
        # Extract data attributes
        self.MAXVAL = data['MAXVAL']
        self.ne = data['ne']
        self.ns = data['ns']
        self.nc = data['nc']
        self.nt = data['nt']
        self.delta = data['delta']
        self.obs = data['obs'].astype(np.float32)  # shape: (ns, nc, nt)
        self.noise_std = data['noise_std']  # shape: (ns, nc)
        self.greens = data['green_tensor'].astype(np.float32)  # shape: (ns, nc, ne, nt)
        # self.greens_fft = fft(self.greens, axis=-1) #fft of greens tensor
        self.omega = 2 * np.pi * fftfreq(self.nt, d=self.delta)

        # Select log-likelihood method
        if method == 'full_mij_uncorrelated':
            self.log_prob = self._log_prob_full_mij_uncorrelated
        elif method == 'full_mij_correlated_exp':
            self.log_prob = self._log_prob_full_mij_correlated_exp
            # self.cov_matrix = exponential_covariance(self.nt, 4)
            # cov_d = np.broadcast_to(self.cov_matrix, (self.ns, self.nc, self.nt, self.nt))
            # self.cov_inv, self.log_cov_det = calc_InversionDeterminant_cd(cov_d)
        else:
            raise ValueError(f"Unknown method: {method}")

    def log_prior(self, m):
        """Check if parameters are within bounds."""
        return 0 if np.all((-self.MAXVAL <= m) & (m <= self.MAXVAL)) else -np.inf

    def _apply_phase_shift(self, pred_fft, shift):
        """Apply phase shift in frequency domain."""
        omega_expanded = self.omega[None, None, :]
        shift_expanded = np.zeros((self.ns, self.nc))
        shift_expanded[:, :2] = shift[::2][:, None]
        shift_expanded[:, 2] = shift[1::2]
        phase_shift = np.exp(-1j * omega_expanded * shift_expanded[:, :, None])
        return np.real(ifft(pred_fft * phase_shift, axis=-1))

    def _log_prob_full_mij_uncorrelated(self, m):
        """Log-likelihood with uncorrelated noise."""
        mij = m[:self.ne]
        amp = (m[self.ne:self.ne + self.ns] + self.MAXVAL) / 720 + 1e-4
        shift = m[self.ne + self.ns:] / 360 #[-10, 10] sec

        noise_amp = self.noise_std * amp[:, None]
        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, mij)
        pred_fft = fft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # noise weighted residuals
        res = (self.obs - pred) / noise_amp[:, :, None]
        lp1 = np.sum(res ** 2)
        lp2 = np.sum(self.nt * 2 * np.log(noise_amp))
        return -0.5 * (lp1 + lp2)

    def _log_prob_full_mij_correlated_exp(self, m):
        """Log-likelihood with correlated noise using exponential covariance."""
        mij = m[:self.ne]
        amp = (m[self.ne:self.ne + self.ns] + self.MAXVAL) / 720 + 1e-4
        shift = m[self.ne + self.ns:] / 360

        noise_amp = self.noise_std * amp[:, None]
        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, mij)
        pred_fft = fft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # covariance matrix weighted residuals
        res = (self.obs - pred) / noise_amp[:, :, None]
        res_cov = np.einsum('...i,...ij', res, cov_inv)
        lp1 = np.einsum('...i,...i', res_cov, res)
        lp1 /= (noise_amp ** 2 * np.exp(2 * log_cov_det / self.nt))
        lp2 = 2 * log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)

    def __call__(self, m):
        """Evaluate log-posterior."""
        lp = self.log_prior(m)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_prob(m) 
