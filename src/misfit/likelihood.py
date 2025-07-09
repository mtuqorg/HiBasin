import os 
import emcee 
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import multiprocessing as mp
from mtuq.util.math import to_mij, to_rho
from mtuq.grid.moment_tensor import to_mt
from mtuq.grid.force import to_force
from mtuq.grid import UnstructuredGrid
from mtuq.misfit.waveform import level2 
from src.util.math import to_lune, Tashiro2MT6,ned2rtp
from src.util.math import exponential_covariance, calc_InversionDeterminant_cd
from src.misfit.misfit_preparation import to_numpy_arrays

os.environ["OMP_NUM_THREADS"] = "1"

## Shared memory for covariance matrix, 
# not a physical variable but a shared resource
shared_data = {}

def pool_initializer(shm_name, shape, dtype_str):
        """Initialize worker for multiprocessing."""
        # Reconnect to shared memory inside each process to calculate the log probability
        dtype = np.dtype(dtype_str)
        existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
        shared_data['cov_inv'] = shared_array
        shared_data['shm'] = existing_shm
        # print(f"[Worker Init] PID={os.getpid()} initialized shared memory: {shm_name} ")

class MCMC_FMT:
    def __init__(self, mistfit_sw, data_sw, greens_sw, noise_std_sw, \
                 cov_inv, log_cov_det, max_noise_parameter=100, M00=None, method='mij_uncorrelated'):
        self.MAXVAL = 3600 # Maximum value for parameters
        # Extract data attributes
        ## self.obs shape: (ns, nc, nt); self.greens shape: (ns, nc, ne, nt)
        self.obs, self.greens = to_numpy_arrays(data_sw, greens_sw)
        if M00 is not None:
            #scale the greens by M00
            self.greens *=  M00

        self.ns, self.nc, self.ne, self.nt = self.greens.shape
        _, self.delta = level2._get_time_sampling(data_sw)

        self.noise_std = noise_std_sw.astype(np.float32)  # shape: (ns, nc)
        self.omega = 2 * np.pi * rfftfreq(self.nt, d=self.delta)

        self.max_noise = max_noise_parameter
        # get time_shfit groups and the min and max time shifts from the mistfit_sw object
        self.time_shift_min = mistfit_sw.time_shift_min
        self.time_shift_max = mistfit_sw.time_shift_max
        self.time_shift_groups = len(mistfit_sw.time_shift_groups)
        assert  self.time_shift_groups in [1,2,3],\
                ValueError("Unsupported number of time shift groups: %d" % len(mistfit_sw.time_shift_groups))
        # Dimensions of the model parameters: ne + ns + ns * shift_group_no (mij, amp, shift)
        self.ndim = self.ne + self.ns + self.ns * self.time_shift_groups

        # Select log-likelihood method
        if method == 'mij_uncorrelated':
            self.log_prob = self._log_prob_full_mij_uncorrelated
        elif method == 'mij_correlated_exp':
            self.log_prob = self._log_prob_full_mij_correlated_exp
        
            # Create the shared memory for the inverse of reference covariance matrix cov_inv
            self.cov_inv_shape = cov_inv.shape
            self.cov_inv_dtype = cov_inv.dtype
            try:
                self.shm = mp.shared_memory.SharedMemory(create=True, size=cov_inv.nbytes)
                ##if using self.shared_array, it will be shared across processes and significantly slow down the computation 
                shared_array = np.ndarray(self.cov_inv_shape, dtype=self.cov_inv_dtype, buffer=self.shm.buf)
                shared_array[:] = cov_inv[:]
            except:
                print("failed to create shared memory")
                if hasattr(self, 'shm'):
                    self.shm.close()
                    self.shm.unlink()
                raise
            self.log_cov_det = log_cov_det
            self.scale = np.exp(2 * self.log_cov_det / self.nt)
        else:
            raise ValueError(f"Unknown method: {method}")

        ##define some constants to speed up the computation
        self.noise_scale1 = self.max_noise / (self.MAXVAL*2)
        self.noise_scale2 = self.noise_scale1 * self.MAXVAL + 1.0e-4 # avoid zero noise amplitude
        self.time_shift_scale1 = (self.time_shift_max - self.time_shift_min) / (self.MAXVAL*2)
        self.time_shift_scale2 = self.time_shift_scale1 * self.MAXVAL + self.time_shift_min 

    def _log_prior(self, m):
        """Define a uninformative (uniform) prior. Check if parameters are within bounds."""
        return 0 if np.all((-self.MAXVAL <= m) & (m <= self.MAXVAL)) else -np.inf

    def _apply_phase_shift(self, pred_fft, shift):
        """Apply phase shift in frequency domain."""
        omega_expanded = self.omega[None, None, :]
        shift_expanded = np.zeros((self.ns, self.nc))
        shift_expanded[:, :2] = shift[::2][:, None]
        shift_expanded[:, 2] = shift[1::2]
        phase_shift = np.exp(-1j * omega_expanded * shift_expanded[:, :, None])
        return np.real(irfft(pred_fft * phase_shift, axis=-1))

    def _log_prob_full_mij_uncorrelated(self, m):
        """Log-likelihood with uncorrelated noise."""
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        mij = m[:self.ne]
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2

        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, mij)
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # noise weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = (self.obs - pred) / noise_amp[:, :, None]
        lp1 = np.sum(res ** 2)
        lp2 = np.sum(self.nt * 2 * np.log(noise_amp))
        return -0.5 * (lp1 + lp2)

    def _log_prob_full_mij_correlated_exp(self, m):
        """Log-likelihood with correlated noise using exponential covariance."""
        if not np.isfinite(self._log_prior(m)): return -np.inf

        mij = m[:self.ne]
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2

        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, mij)
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # covariance matrix weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = (self.obs - pred) / noise_amp[:, :, None]
        lp1 = np.einsum('sct, sctt, sct->sc', res, shared_data['cov_inv'], res) 
        lp1 /= (noise_amp ** 2 * self.scale)

        lp2 = 2 * self.log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)

    def get_sampler(self, method='emcee', nchains=512):
        """Get the sampler for MCMC."""
        ## forking with shared memory is more stable than spawning
        ctx = mp.get_context("fork")
        pool = ctx.Pool(initializer=pool_initializer, initargs=(self.shm.name, self.cov_inv_shape, self.cov_inv_dtype.str))
        if method == 'emcee':
            # Use emcee's EnsembleSampler
            sampler = emcee.EnsembleSampler(nchains, self.ndim, self.log_prob, pool=pool)
            print('Creating a emcee sampler with chains=%s and ndim=%s' % (nchains, self.ndim))
        else:
            raise ValueError(f"Unknown method: {method}")
        return sampler, pool
    
    def cleanup(self, pool):
        """Cleanup shared memory."""
        pool.close()
        pool.join()
        self.shm.close()
        self.shm.unlink()
        print("Shared memory cleaned up.") 

    def get_solution(self, emcee_sampler, warm_up_steps, thin, source_type='full'):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print ('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)
 
        ##transformed MT parameters from Lune to premitive parameterization
        if source_type=='full':
            v = m_sol[0] / 10800
            w = m_sol[1] * np.pi / 9600             
            kappa = (m_sol[2] + self.MAXVAL) / 20           ##0, 360
            sigma = m_sol[3] / 40                           ##-90, 90
            h = (m_sol[4] + self.MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[5]+self.MAXVAL)/3600 + 4)   ##Mw: 4-6
        elif source_type=='dc':
            v = 0
            w = 0            
            kappa = (m_sol[0] + self.MAXVAL) / 20           ##0, 360
            sigma = m_sol[1] / 40                           ##-90, 90
            h = (m_sol[2] + self.MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[3]+self.MAXVAL)/3600 + 4)   ##Mw: 4-6
        elif source_type=='deviatoric':
            v = m_sol[0] / 10800
            w = 0          
            kappa = (m_sol[1] + self.MAXVAL) / 20           ##0, 360
            sigma = m_sol[2] / 40                           ##-90, 90
            h = (m_sol[3] + self.MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[4]+self.MAXVAL)/3600 + 4)   ##Mw: 4-6
        elif source_type=='force':
            phi = (m_sol[0]+ self.MAXVAL) / 20   #[0, 360]
            h = m_sol[1] / 3600                  #[-1,1]
            F0 = (m_sol[2] + self.MAXVAL)        #[0,7200]
        elif source_type == 'mij':
            m_sol *= 10**15
            rho,v,w,kappa,sigma,h = to_lune(m_sol[0:6])
        else:
            #source type = Tashiro
            m_sol[:5] = (self.MAXVAL + m_sol[:5]) / 7200   #(0,1)
            m_sol[5] = (self.MAXVAL + m_sol[5]) / 3600 + 4 #Mw 4-6
            mij = Tashiro2MT6(m_sol[:6])
            mij = ned2rtp(mij) #up-south-east convention
            rho,v,w,kappa,sigma,h = to_lune(mij)
            
        if source_type=='force':
            solution = UnstructuredGrid(
                dims=('F0','phi', 'h'),
                coords=(F0, phi, h),
                callback=to_force)   
        else:
            ## moment tensor
            solution = UnstructuredGrid(
                dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
                coords=(rho, v,w,kappa,sigma,h),
                callback=to_mt)
        return solution

 
