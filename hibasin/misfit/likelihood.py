import os
import emcee
import numpy as np
import numpy.ma as ma
from numpy.fft import rfft, irfft, rfftfreq
import multiprocessing as mp
from mtuq.util.math import to_mij, to_rho, to_rtp
from mtuq.grid.moment_tensor import to_mt
from mtuq.grid.force import to_force
from mtuq.grid import UnstructuredGrid
from mtuq.misfit.waveform import level2
from hibasin.util.math import to_lune, Tashiro2MT6, ned2rtp
from hibasin.util.math import exponential_covariance, calc_InversionDeterminant_cd
from hibasin.misfit.misfit_preparation import to_numpy_arrays, get_timeshift_mask

os.environ["OMP_NUM_THREADS"] = "1"

# Shared memory for covariance matrix across worker processes
shared_data = {}


def pool_initializer(shm_name, shape, dtype_str):
    dtype = np.dtype(dtype_str)
    existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    shared_data['cov_inv'] = shared_array
    shared_data['shm'] = existing_shm


class _MCMC_Base:
    """
    Base class for MCMC solvers.

    Handles data loading, likelihood computation (correlated/uncorrelated),
    shared memory management, MCMC sampling, and diagnostics.

    Subclasses must implement:
        _params_to_mij(m)  – convert sampled vector to source parameters
        get_solution(...)  – extract physical source solution from chains
        save_chains(...)   – save and transform sampled chains to disk
    """

    _apply_m00 = False  # set True in subclasses that pre-scale greens by M00

    def __init__(self, mistfit_sw, data_sw, greens_sw, noise_std_sw,
                 cov_inv=None, log_cov_det=None, max_noise_parameter=100,
                 M00=None, noise_type='uncorrelated'):
        self.MAXVAL = 3600

        # obs shape: (ns, nc, nt); greens shape: (ns, nc, ne, nt)
        self.obs, self.greens, self.weight_mask = to_numpy_arrays(data_sw, greens_sw)

        if M00 is not None and self._apply_m00:
            self.M00 = M00
            self.greens *= self.M00
        else:
            self.M00 = None

        self.ns, self.nc, self.ne, self.nt = self.greens.shape
        _, self.delta = level2._get_time_sampling(data_sw)

        self.noise_std = noise_std_sw.astype(np.float32)
        self.omega = 2 * np.pi * rfftfreq(self.nt, d=self.delta)

        self.max_noise = max_noise_parameter
        self.time_shift_min = mistfit_sw.time_shift_min
        self.time_shift_max = mistfit_sw.time_shift_max
        self.time_shift_groups = len(mistfit_sw.time_shift_groups)
        assert self.time_shift_groups in [1, 2], \
            "Unsupported number of time shift groups: %d" % self.time_shift_groups

        self.timeshift_mask = get_timeshift_mask(self.weight_mask, self.time_shift_groups)

        self.noise_type = noise_type
        if noise_type == 'correlated':
            self.cov_inv_shape = cov_inv.shape
            self.cov_inv_dtype = cov_inv.dtype
            try:
                self.shm = mp.shared_memory.SharedMemory(create=True, size=cov_inv.nbytes)
                shared_array = np.ndarray(self.cov_inv_shape, dtype=self.cov_inv_dtype,
                                          buffer=self.shm.buf)
                shared_array[:] = cov_inv[:]
            except Exception:
                print("failed to create shared memory")
                if hasattr(self, 'shm'):
                    self.shm.close()
                    self.shm.unlink()
                raise
            self.log_cov_det = log_cov_det
            self.scale = np.exp(2 * self.log_cov_det / self.nt)

        self.MAXVAL2 = self.MAXVAL * 2
        self.noise_scale1 = self.max_noise / self.MAXVAL2
        self.noise_scale2 = self.noise_scale1 * self.MAXVAL
        self.time_shift_scale1 = (self.time_shift_max - self.time_shift_min) / self.MAXVAL2
        self.time_shift_scale2 = self.time_shift_scale1 * self.MAXVAL + self.time_shift_min

        self.ndim = self.ne + self.ns + np.sum(self.timeshift_mask)

        # Bind likelihood function once to avoid per-call string comparison
        self._likelihood_fn = (self._uncorrelated_likelihood
                               if noise_type == 'uncorrelated'
                               else self._correlated_likelihood)

    # ------------------------------------------------------------------
    # Prior and phase shift (shared across all parameterizations)
    # ------------------------------------------------------------------

    def _log_prior(self, m):
        return 0 if np.all((-self.MAXVAL <= m) & (m <= self.MAXVAL)) else -np.inf

    def _apply_phase_shift(self, pred_fft, shift):
        omega_expanded = self.omega[None, None, :]
        shift_expanded = np.zeros((self.ns, self.nc))
        if self.time_shift_groups == 1:
            shift_expanded[:, :] = shift[:, None]
        else:
            shift_expanded[:, :2] = shift[::2][:, None]
            shift_expanded[:, 2] = shift[1::2]
        phase_shift = np.exp(-1j * omega_expanded * shift_expanded[:, :, None])
        return np.real(irfft(pred_fft * phase_shift, axis=-1))

    def _decode_amp_shift(self, m):
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2
        shift = np.zeros(self.time_shift_groups * self.ns)
        shift[self.timeshift_mask] = (m[self.ne + self.ns:] * self.time_shift_scale1
                                      + self.time_shift_scale2)
        return amp, shift

    # ------------------------------------------------------------------
    # Parameterization hook (override in subclasses)
    # ------------------------------------------------------------------

    def _params_to_mij(self, m):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Likelihood functions
    # ------------------------------------------------------------------

    def log_prob(self, m):
        if not np.isfinite(self._log_prior(m)):
            return -np.inf
        mij = self._params_to_mij(m)
        amp, shift = self._decode_amp_shift(m)
        pred = np.einsum('scet,e->sct', self.greens, mij)
        pred = self._apply_phase_shift(rfft(pred, axis=-1), shift)
        return self._likelihood_fn(pred, amp)

    def _uncorrelated_likelihood(self, pred, amp):
        noise_amp = self.noise_std * amp[:, None]
        res = (self.obs - pred) / noise_amp[:, :, None]
        res = ma.masked_array(res, np.broadcast_to(self.weight_mask[:, :, None], res.shape))
        noise_amp = ma.masked_array(noise_amp, self.weight_mask)
        lp1 = np.sum(res ** 2)
        lp2 = np.sum(self.nt * 2 * np.log(noise_amp))
        return -0.5 * (lp1 + lp2)

    def _correlated_likelihood(self, pred, amp):
        noise_amp = self.noise_std * amp[:, None]
        res = self.obs - pred
        lp1 = np.einsum('sct, sctt, sct->sc', res, shared_data['cov_inv'], res)
        lp1 /= (noise_amp ** 2 * self.scale)
        lp2 = 2 * self.log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)

    # ------------------------------------------------------------------
    # Sampler, cleanup, diagnostics
    # ------------------------------------------------------------------

    def get_sampler(self, method='emcee', nchains=512):
        ctx = mp.get_context("fork")
        if self.noise_type == 'correlated':
            pool = ctx.Pool(initializer=pool_initializer,
                            initargs=(self.shm.name, self.cov_inv_shape,
                                      self.cov_inv_dtype.str))
        else:
            pool = ctx.Pool()
        if method == 'emcee':
            sampler = emcee.EnsembleSampler(nchains, self.ndim, self.log_prob, pool=pool)
            print('Creating a emcee sampler with chains=%s and ndim=%s' % (nchains, self.ndim))
        else:
            raise ValueError(f"Unknown sampler method: {method}")
        return sampler, pool

    def reset(self):
        if self.M00 is not None:
            self.greens /= self.M00

    def cleanup(self, pool):
        pool.close()
        pool.join()
        if self.noise_type == 'correlated':
            self.shm.close()
            self.shm.unlink()
            print("Shared memory cleaned up.")
        self.reset()

    def diagnose(self, sampler):
        tau = sampler.get_autocorr_time(tol=0)
        print('\nAutocorrelation time for each coordinates of the model space:\n    ', tau)

    # ------------------------------------------------------------------
    # Shared helpers for subclass get_solution / save_chains
    # ------------------------------------------------------------------

    def _noise_timeshift_solution(self, m_sol):
        noise = m_sol[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2
        tau = m_sol[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2
        if self.time_shift_groups == 1:
            return noise, np.repeat(tau, 2)
        tau_full = np.zeros(2 * self.ns)
        tau_full[self.timeshift_mask] = tau
        return noise, tau_full

    def _transform_noise_timeshift_chains(self, samples):
        samples[:, self.ne:self.ne + self.ns] = (
            samples[:, self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2)
        samples[:, self.ne + self.ns:] = (
            samples[:, self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2)
        return samples

    def _save_chains_core(self, sampler, flat_samples, file_path, tag, thin):
        self.chain_fname = file_path + 'MCMC_sampling_%s_%s_model.npy' % (tag, self.noise_type)
        np.save(self.chain_fname, flat_samples)
        
        self.logprob_fname = file_path + 'MCMC_sampling_%s_%s_log_prob.npy' % (tag, self.noise_type)
        log_prob = sampler.get_log_prob(discard=0, thin=thin, flat=True)
        np.save(self.logprob_fname, log_prob)

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        raise NotImplementedError

    def save_chains(self, sampler, file_path='./', thin=1):
        raise NotImplementedError


# =============================================================================
# Moment tensor parameterizations
# =============================================================================

class MCMC_FullMij(_MCMC_Base):
    """
    MCMC solver using full moment tensor (mij) parameterization (ne=6).
    Green's functions are pre-scaled by M00.
    """

    _apply_m00 = True

    def _params_to_mij(self, m):
        return m[:6]

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        rho, v, w, kappa, sigma, h = to_lune(m_sol[:6] * self.M00)

        source_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)
        noise_sol, tau_sol = self._noise_timeshift_solution(m_sol)
        return source_solution, noise_sol, tau_sol

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        flat = self._transform_noise_timeshift_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'mij', thin)


class MCMC_DeviatoricMij(_MCMC_Base):
    """
    MCMC solver using deviatoric moment tensor (mij) parameterization (ne=5).
    Mrr is constrained as -(Mtt + Mpp); Green's functions are pre-scaled by M00.
    """

    _apply_m00 = True

    def __init__(self, mistfit_sw, data_sw, greens_sw, noise_std_sw,
                 cov_inv=None, log_cov_det=None, max_noise_parameter=100,
                 M00=None, noise_type='uncorrelated'):
        super().__init__(mistfit_sw, data_sw, greens_sw, noise_std_sw,
                         cov_inv=cov_inv, log_cov_det=log_cov_det,
                         max_noise_parameter=max_noise_parameter,
                         M00=M00, noise_type=noise_type)
        # Override ne/ndim: deviatoric samples 5 MT params, not 6
        self.ne = 5
        self.ndim = self.ne + self.ns + np.sum(self.timeshift_mask)
        self._likelihood_fn = (self._uncorrelated_likelihood
                               if noise_type == 'uncorrelated'
                               else self._correlated_likelihood)

    def _params_to_mij(self, m):
        mij = np.empty(6, dtype=float)
        mij[0] = -m[0] - m[1]  # Mrr = -(Mtt + Mpp)
        mij[1:] = m[:5]
        return mij

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        mij5 = m_sol[:5] * self.M00
        mij = np.array([-mij5[0] - mij5[1],
                         mij5[0], mij5[1], mij5[2], mij5[3], mij5[4]])
        rho, v, w, kappa, sigma, h = to_lune(mij)

        source_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)
        noise_sol, tau_sol = self._noise_timeshift_solution(m_sol)
        return source_solution, noise_sol, tau_sol

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        flat = self._transform_noise_timeshift_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'mij_deviatoric', thin)


class MCMC_TT2015(_MCMC_Base):
    """
    MCMC solver using Tape & Tape (2015) lune parameterization.

    Sampled parameters (ne=6): v, w, kappa, sigma, h, Mw (encoded).
    Also supports deviatoric (ne=5, w fixed to 0) and DC (ne=4, v=w=0)
    in get_solution and save_chains for chains produced by external means.
    """

    def _params_to_mij(self, m):
        v = m[0] / 10800
        w = m[1] * np.pi / 9600
        kappa = (m[2] + self.MAXVAL) / 20       # [0, 360]
        sigma = m[3] / 40                        # [-90, 90]
        h = (m[4] + self.MAXVAL) / 7200          # cos(dip)
        rho = to_rho((m[5] + self.MAXVAL) / 3600 + 4)  # Mw 4-6
        return to_mij(rho, v, w, kappa, sigma, h)

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        if self.ne == 6:  # full MT
            v = m_sol[0] / 10800
            w = m_sol[1] * np.pi / 9600
            kappa = (m_sol[2] + self.MAXVAL) / 20
            sigma = m_sol[3] / 40
            h = (m_sol[4] + self.MAXVAL) / 7200
            rho = to_rho((m_sol[5] + self.MAXVAL) / 3600 + 4)
        elif self.ne == 5:  # deviatoric (w=0)
            v = m_sol[0] / 10800
            w = 0
            kappa = (m_sol[1] + self.MAXVAL) / 20
            sigma = m_sol[2] / 40
            h = (m_sol[3] + self.MAXVAL) / 7200
            rho = to_rho((m_sol[4] + self.MAXVAL) / 3600 + 4)
        else:  # DC (v=w=0), ne=4
            v = 0
            w = 0
            kappa = (m_sol[0] + self.MAXVAL) / 20
            sigma = m_sol[1] / 40
            h = (m_sol[2] + self.MAXVAL) / 7200
            rho = to_rho((m_sol[3] + self.MAXVAL) / 3600 + 4)

        source_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)
        noise_sol, tau_sol = self._noise_timeshift_solution(m_sol)
        return source_solution, noise_sol, tau_sol

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        to_rho_vec = np.vectorize(to_rho)

        if self.ne == 6:
            flat[:, :6] = np.column_stack((
                flat[:, 0] / 10800,
                flat[:, 1] * np.pi / 9600,
                (flat[:, 2] + self.MAXVAL) / 20,
                flat[:, 3] / 40,
                (flat[:, 4] + self.MAXVAL) / 7200,
                to_rho_vec((flat[:, 5] + self.MAXVAL) / 3600 + 4)
            ))
        elif self.ne == 5:
            flat[:, :5] = np.column_stack((
                flat[:, 0] / 10800,
                (flat[:, 1] + self.MAXVAL) / 20,
                flat[:, 2] / 40,
                (flat[:, 3] + self.MAXVAL) / 7200,
                to_rho_vec((flat[:, 4] + self.MAXVAL) / 3600 + 4)
            ))
        elif self.ne == 4:
            flat[:, :4] = np.column_stack((
                (flat[:, 0] + self.MAXVAL) / 20,
                flat[:, 1] / 40,
                (flat[:, 2] + self.MAXVAL) / 7200,
                to_rho_vec((flat[:, 3] + self.MAXVAL) / 3600 + 4)
            ))

        flat = self._transform_noise_timeshift_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'tt2015', thin)


class MCMC_Tashiro(_MCMC_Base):
    """
    MCMC solver using Tashiro parameterization for full moment tensor inversion.

    Sampled parameters (ne=6): x1–x5 in (-MAXVAL, MAXVAL) mapped to (0,1),
    and x6 encoding Mw in (4, 6).
    """

    def _params_to_mij(self, m):
        mc = m[:6].copy()
        mc[:5] = (mc[:5] + self.MAXVAL) / self.MAXVAL2  # x_i -> (0, 1)
        mc[5] = (mc[5] + self.MAXVAL) / self.MAXVAL + 4  # Mw 4-6
        return ned2rtp(Tashiro2MT6(mc))

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        xi = (self.MAXVAL + m_sol[:5]) / self.MAXVAL2       # (0, 1)
        mw = (self.MAXVAL + m_sol[5]) / self.MAXVAL + 4     # Mw 4-6
        mij = ned2rtp(Tashiro2MT6(np.concatenate([xi, [mw]])))
        rho, v, w, kappa, sigma, h = to_lune(mij)

        source_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)
        noise_sol, tau_sol = self._noise_timeshift_solution(m_sol)
        return source_solution, noise_sol, tau_sol

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        flat[:, :6] = np.column_stack((
            (self.MAXVAL + flat[:, :5]) / self.MAXVAL2,    # x_i -> (0, 1)
            (self.MAXVAL + flat[:, 5]) / self.MAXVAL + 4   # Mw 4-6
        ))
        flat = self._transform_noise_timeshift_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'tashiro', thin)


# =============================================================================
# Single force and joint MT + force
# =============================================================================

class MCMC_Force(_MCMC_Base):
    """
    MCMC solver for single force inversion.

    Sampled parameters (ne=3): phi, h, F0 (encoded).
    Green's functions are pre-scaled by M00.
    """

    _apply_m00 = True

    def _params_to_mij(self, m):
        phi = (m[0] + self.MAXVAL) / 20    # [0, 360]
        h = m[1] / self.MAXVAL             # [-1, 1]
        F0 = m[2] + self.MAXVAL            # [0, 7200]
        return to_rtp(F0, phi, h)

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        phi = (m_sol[0] + self.MAXVAL) / 20
        h = m_sol[1] / self.MAXVAL
        F0 = m_sol[2] + self.MAXVAL

        source_solution = UnstructuredGrid(
            dims=('F0', 'phi', 'h'),
            coords=(F0, phi, h),
            callback=to_force)
        noise_sol, tau_sol = self._noise_timeshift_solution(m_sol)
        return source_solution, noise_sol, tau_sol

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        flat[:, :3] = np.column_stack((
            (flat[:, 0] + self.MAXVAL) / 20,
            flat[:, 1] / self.MAXVAL,
            flat[:, 2] + self.MAXVAL
        ))
        flat = self._transform_noise_timeshift_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'force', thin)


class MCMC_MTSF(_MCMC_Base):
    """
    MCMC solver for joint moment tensor + single force inversion.

    Parameter vector layout (ne=9 from greens shape):
        m[0:6]     – MT components (mij, in up-south-east convention)
        m[6:9]     – force parameters (phi, h, F0 encoded)
        m[9:9+ns]  – per-station noise amplitudes
        m[9+ns:]   – time shifts
    Green's functions must have ne=9 (6 MT + 3 force elementary seismograms)
    and are pre-scaled by M00.
    """

    _apply_m00 = True

    def _params_to_mij(self, m):
        mij = m[:6].copy()
        phi = (m[6] + self.MAXVAL) / 20    # [0, 360]
        h = m[7] / self.MAXVAL             # [-1, 1]
        F0 = m[8] + self.MAXVAL            # [0, 7200]
        fij = to_rtp(F0, phi, h)
        return np.concatenate((mij, fij))

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        mij = m_sol[:6] * self.M00
        rho, v, w, kappa, sigma, h_mt = to_lune(mij)
        mt_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h_mt),
            callback=to_mt)

        phi = (m_sol[6] + self.MAXVAL) / 20
        h_f = m_sol[7] / self.MAXVAL
        F0 = m_sol[8] + self.MAXVAL
        force_solution = UnstructuredGrid(
            dims=('F0', 'phi', 'h'),
            coords=(F0, phi, h_f),
            callback=to_force)

        noise_sol, tau_sol = self._noise_timeshift_solution(m_sol)
        return mt_solution, force_solution, noise_sol, tau_sol

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        # MT part (indices 0-5): no transform needed (already in mij units)
        # Force part (indices 6-8): convert to physical units
        flat[:, 6:9] = np.column_stack((
            (flat[:, 6] + self.MAXVAL) / 20,
            flat[:, 7] / self.MAXVAL,
            flat[:, 8] + self.MAXVAL
        ))
        flat = self._transform_noise_timeshift_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'mtsf', thin)
