import os 
import emcee 
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import multiprocessing as mp
from mtuq.util import asarray
from mtuq.util.math import to_mij, to_rho, to_rtp
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

class MCMC_SOLVER:
    def __init__(self, mistfit_sw, data_sw, greens_sw, noise_std_sw, \
                 cov_inv=None, log_cov_det=None, max_noise_parameter=100, M00=None, method='mij_uncorrelated'):
        self.MAXVAL = 3600 # Maximum value for parameters
        # Extract data attributes
        ## self.obs shape: (ns, nc, nt); self.greens shape: (ns, nc, ne, nt)
        #Note: the greens and mij are in up-south-east convention
        self.obs, self.greens = to_numpy_arrays(data_sw, greens_sw)

        ##debug
        # from netCDF4 import Dataset
        # dataset_fname = '/Users/u7091895/Documents/Research/DPRK_data/Data_DPRK2017_velMDJ2_depth500m.nc4'
        # rootgrp = Dataset(dataset_fname)
        # # self.obs = np.array(rootgrp.variables['obs_data'][:])        
        # # gf = np.array(rootgrp.variables['greens_tensor'][:])   
        # noise_std_sw = np.array(rootgrp.variables['noise_std'][:])  
        # rootgrp.close()
        #convert the GF to rtp
        # for s in range(7):
        #     self.greens[s,:,0] = gf[2][s]
        #     self.greens[s,:,1] = gf[0][s]
        #     self.greens[s,:,2] = gf[1][s]
        #     self.greens[s,:,3] = gf[4][s]
        #     self.greens[s,:,4] = -1*gf[5][s]
        #     self.greens[s,:,5] = -1*gf[3][s]
            # Mrr=Mzz 2
            # Mtt=Mxx 0
            # Mpp=Myy 1
            # Mrt=Mxz 4
            # Mrp=-Myz 5
            # Mtp=-Mxy 3
        ###

        if M00 is not None and method.split("_")[0] in ['mij', 'force', 'mtsf']:
            #scale the greens by M00 only used for mij, force, or joint inversion
            self.M00 = M00
            self.greens *=  self.M00
        else:
            self.M00 = None

        self.ns, self.nc, self.ne, self.nt = self.greens.shape
        _, self.delta = level2._get_time_sampling(data_sw)

        self.noise_std = noise_std_sw.astype(np.float32)  # shape: (ns, nc)
        self.omega = 2 * np.pi * rfftfreq(self.nt, d=self.delta)

        self.max_noise = max_noise_parameter
        # get time_shfit groups and the min and max time shifts from the mistfit_sw object
        self.time_shift_min = mistfit_sw.time_shift_min
        self.time_shift_max = mistfit_sw.time_shift_max
        self.time_shift_groups = len(mistfit_sw.time_shift_groups)
        assert  self.time_shift_groups in [1,2],\
                ValueError("Unsupported number of time shift groups: %d" % len(mistfit_sw.time_shift_groups))
        # Dimensions of the model parameters: ne + ns + ns * shift_group_no (mij, amp, shift)
        self.ndim = self.ne + self.ns + self.ns * self.time_shift_groups

        # Select log-likelihood method
        self.method = method
        if method == 'mij_uncorrelated':
            self.log_prob = self._log_prob_full_mij_uncorrelated
        elif method == 'tt2015_uncorrelated':
            self.log_prob = self._log_prob_full_tt2015_uncorrelated
        elif method == 'tashiro_uncorrelated':
            self.log_prob = self._log_prob_full_tashiro_uncorrelated
        elif method == 'force_uncorrelated':
            self.log_prob = self._log_prob_force_uncorrelated   
        elif method == 'mtsf_uncorrelated':
            self.log_prob = self._log_prob_mtsf_uncorrelated
        else:##correlated data noise        
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

            # Select the log probability for correlated data noise treatment
            if method == 'mij_correlated':
                self.log_prob = self._log_prob_full_mij_correlated
            elif method == 'tt2015_correlated':
                self.log_prob = self._log_prob_full_tt2015_correlated
            elif method == 'tashiro_correlated':
                self.log_prob = self._log_prob_full_tashiro_correlated
            elif method == 'force_correlated':
                self.log_prob = self._log_prob_force_correlated
            elif method == 'mtsf_correlated':
                self.log_prob = self._log_prob_mtsf_correlated
            else:
                raise ValueError(f"Unknown method: {method}")

        ##define some constants to speed up the computation
        self.MAXVAL2 = self.MAXVAL * 2
        self.noise_scale1 = self.max_noise / self.MAXVAL2
        self.noise_scale2 = self.noise_scale1 * self.MAXVAL #+ 1.0e-4 # avoid zero noise amplitude
        self.time_shift_scale1 = (self.time_shift_max - self.time_shift_min) / self.MAXVAL2
        self.time_shift_scale2 = self.time_shift_scale1 * self.MAXVAL + self.time_shift_min 

    def _log_prior(self, m):
        """Define a uninformative (uniform) prior. Check if parameters are within bounds."""
        return 0 if np.all((-self.MAXVAL <= m) & (m <= self.MAXVAL)) else -np.inf

    def _apply_phase_shift(self, pred_fft, shift):
        """Apply phase shift in frequency domain."""
        omega_expanded = self.omega[None, None, :]
        shift_expanded = np.zeros((self.ns, self.nc))
        if self.time_shift_groups == 1:
            shift_expanded[:, :] = shift[:, None]
        else:
            shift_expanded[:, :2] = shift[::2][:, None]
            shift_expanded[:, 2] = shift[1::2]
        phase_shift = np.exp(-1j * omega_expanded * shift_expanded[:, :, None])
        return np.real(irfft(pred_fft * phase_shift, axis=-1))

    def _log_prob_full_mij_uncorrelated(self, m):
        """Full MT Inversion with treatment of uncorrelated data noise and time shifts as free parameters using mij parameterization"""
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        #mij in up-south-east
        mij = m[:self.ne]
        #station-based noise
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        #station-based time shift
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

    def _log_prob_full_tt2015_uncorrelated(self, m):
        '''
        Full MT Inversion with treatment of uncorrelated data noise and time shifts as free parameters using TT2015 parameterization
        '''
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        ##convert sampled lune paramters to mij as transformed parameters in Bayesian sampling
        v = m[0] / 10800
        w = m[1] * np.pi / 9600             
        kappa = (m[2] + self.MAXVAL) / 20             ##0, 360
        sigma = m[3] / 40                             ##-90, 90
        h = (m[4] + self.MAXVAL) / 7200               ##cos(dip)
        rho = to_rho((m[5]+self.MAXVAL)/3600 + 4)     ##Mw: 4-6 for small-to-moderate events
    
        mij = to_mij(rho,v,w,kappa,sigma,h) #in up-south-east convention
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
    
    def _log_prob_full_tashiro_uncorrelated(self, m):
        """Full MT Inversion with treatment of uncorrelated data noise and time shifts as free parameters using Tashiro MT parameterization"""
        if not np.isfinite(self._log_prior(m)): return -np.inf

        ##use the primitive MT in Bayesian sampling
        m[:5] = ( m[:5]+self.MAXVAL) / self.MAXVAL2     #x_i (1-5) ~ (0,1)
        m[5] = (m[5]+self.MAXVAL)/self.MAXVAL + 4       #Mw 4-6
        mij = ned2rtp(Tashiro2MT6(m[:6])) #in up-south-east convention
        
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
    
    def _log_prob_full_tt2015_correlated(self, m):
        '''
        Full MT Inversion with treatment of correlated data noise and time shifts as free parameters using TT2015 parameterization
        '''
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        ##convert sampled lune paramters to mij as transformed parameters in Bayesian sampling
        v = m[0] / 10800
        w = m[1] * np.pi / 9600             
        kappa = (m[2] + self.MAXVAL) / 20             ##0, 360
        sigma = m[3] / 40                             ##-90, 90
        h = (m[4] + self.MAXVAL) / 7200               ##cos(dip)
        rho = to_rho((m[5]+self.MAXVAL)/3600 + 4)     ##Mw: 4-6 for small-to-moderate events
    
        mij = to_mij(rho,v,w,kappa,sigma,h) #in up-south-east convention
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2

        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, mij)
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # covariance matrix weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = self.obs - pred
        lp1 = np.einsum('sct, sctt, sct->sc', res, shared_data['cov_inv'], res) 
        lp1 /= (noise_amp ** 2 * self.scale)

        lp2 = 2 * self.log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)

    def _log_prob_full_mij_correlated(self, m):
        """Full MT Inversion with treatment of correlated data noise and time shifts as free parameters using mij parameterization"""
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
        res = self.obs - pred
        lp1 = np.einsum('sct, sctt, sct->sc', res, shared_data['cov_inv'], res) 
        lp1 /= (noise_amp ** 2 * self.scale)

        lp2 = 2 * self.log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)
    
    def _log_prob_full_tashiro_correlated(self, m):
        """Full MT Inversion with treatment of correlated data noise and time shifts as free parameters using Tashiro MT parameterization"""
        if not np.isfinite(self._log_prior(m)): return -np.inf

        ##use the primitive MT in Bayesian sampling
        m[:5] = ( m[:5]+self.MAXVAL) / self.MAXVAL2     #x_i (1-5) ~ (0,1)
        m[5] = (m[5]+self.MAXVAL)/self.MAXVAL + 4       #Mw 4-6
        mij = ned2rtp(Tashiro2MT6(m[:6]) ) #in up-south-east convention
        
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2
        
        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, mij)
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # covariance matrix weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = self.obs - pred
        lp1 = np.einsum('sct, sctt, sct->sc', res, shared_data['cov_inv'], res) 
        lp1 /= (noise_amp ** 2 * self.scale)

        lp2 = 2 * self.log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)

    def _log_prob_force_uncorrelated(self, m):
        """Single force inversion with treatment of uncorrelated data noise and time shifts as free parameters"""
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        phi = (m[0]+ self.MAXVAL) / 20   #[0, 360]
        h = m[1] / self.MAXVAL           #[-1,1]
        F0 = m[2] + self.MAXVAL          #[0, 7200]
        fij = to_rtp(F0, phi, h)

        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2

        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, fij)
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # noise weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = (self.obs - pred) / noise_amp[:, :, None]
        lp1 = np.sum(res ** 2)
        lp2 = np.sum(self.nt * 2 * np.log(noise_amp))
        return -0.5 * (lp1 + lp2)
    
    def _log_prob_force_correlated(self, m):
        """Single force inversion with treatment of correlated data noise and time shifts as free parameters"""
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        phi = (m[0]+ self.MAXVAL) / 20   #[0, 360]
        h = m[1] / self.MAXVAL           #[-1,1]
        F0 = m[2] + self.MAXVAL          #[0, 7200]
        fij = to_rtp(F0, phi, h)

        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2

        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, fij)
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # covariance matrix weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = self.obs - pred
        lp1 = np.einsum('sct, sctt, sct->sc', res, shared_data['cov_inv'], res) 
        lp1 /= (noise_amp ** 2 * self.scale)

        lp2 = 2 * self.log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)
    
    def _log_prob_mtsf_uncorrelated(self, m):
        '''
        Joint inversion of single force inversion and full moment tensor
        with treatment of uncorrelated data noise and time shifts as free parameters
        Here the MT parameters are in the primitive parameterization
        '''
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        #Moment tensor
        mij = m[:self.ne]
        #Force
        phi = (m[self.ne]+ self.MAXVAL) / 20    #[0, 360]
        h = m[self.ne+1] / self.MAXVAL          #[-1,1]
        F0 = m[self.ne+2] + self.MAXVAL         #[0, 7200]
        fij = to_rtp(F0, phi, h)
        
        #Amplitude and time shift
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2
        
        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, np.concatenate((mij,fij)))
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # noise weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = (self.obs - pred) / noise_amp[:, :, None]
        lp1 = np.sum(res ** 2)
        lp2 = np.sum(self.nt * 2 * np.log(noise_amp))
        return -0.5 * (lp1 + lp2)
    
    def _log_prob_mtsf_correlated(self, m):
        '''
        Joint inversion of single force inversion and full moment tensor
        with treatment of correlated data noise and time shifts as free parameters
        Here the MT parameters are in the primitive parameterization
        '''
        if not np.isfinite(self._log_prior(m)): return -np.inf
        
        #Moment tensor
        mij = m[:self.ne]
        #Force
        phi = (m[self.ne]+ self.MAXVAL) / 20    #[0, 360]
        h = m[self.ne+1] / self.MAXVAL          #[-1,1]
        F0 = m[self.ne+2] + self.MAXVAL         #[0, 7200]
        fij = to_rtp(F0, phi, h)
        
        #Amplitude and time shift
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2 
        shift = m[self.ne + self.ns:] * self.time_shift_scale1 + self.time_shift_scale2
        
        # calculate predicted waveforms: d=Gm
        pred = np.einsum('scet,e->sct', self.greens, np.concatenate((mij,fij)))
        # apply phase shift in frequency domain based on time shifts
        pred_fft = rfft(pred, axis=-1)
        pred = self._apply_phase_shift(pred_fft, shift)

        # covariance matrix weighted residuals
        noise_amp = self.noise_std * amp[:, None]
        res = self.obs - pred
        lp1 = np.einsum('sct, sctt, sct->sc', res, shared_data['cov_inv'], res) 
        lp1 /= (noise_amp ** 2 * self.scale)

        lp2 = 2 * self.log_cov_det + 2 * self.nt * np.log(noise_amp)
        return -0.5 * np.sum(lp1 + lp2)

    def get_sampler(self, method='emcee', nchains=512):
        """Get the sampler for MCMC."""
        ## forking with shared memory is more stable than spawning
        ctx = mp.get_context("fork")
        if self.method.split("_")[-1] == 'correlated':
            pool = ctx.Pool(initializer=pool_initializer, initargs=(self.shm.name, self.cov_inv_shape, self.cov_inv_dtype.str))
        else:
            pool = ctx.Pool()
        if method == 'emcee':
            # Use emcee's EnsembleSampler
            sampler = emcee.EnsembleSampler(nchains, self.ndim, self.log_prob, pool=pool)
            print('Creating a emcee sampler with chains=%s and ndim=%s' % (nchains, self.ndim))
        else:
            raise ValueError(f"Unknown method: {method}")
        return sampler, pool
    
    def reset(self):
        if self.M00 is not None and 'mij' == self.method.split("_")[0]:
            #scale back the greens by M00 only used for mij
            self.greens /=  self.M00
    
    def cleanup(self, pool):
        """Cleanup shared memory."""
        pool.close()
        pool.join()
        if self.method.split("_")[-1] == 'correlated':
            self.shm.close()
            self.shm.unlink()
            print("Shared memory cleaned up.") 
        self.reset()

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print ('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)
 
        ##transformed MT parameters to mtuq UnstructuredGrid
        source_type = self.method.split('_')[0]  # e.g., 'tt2015', 'mij', 'tashiro', 'force'
        if source_type=='tt2015' and self.ne == 6: #full MT
            v = m_sol[0] / 10800
            w = m_sol[1] * np.pi / 9600             
            kappa = (m_sol[2] + self.MAXVAL) / 20           ##0, 360
            sigma = m_sol[3] / 40                           ##-90, 90
            h = (m_sol[4] + self.MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[5]+self.MAXVAL)/3600 + 4)   ##Mw: 4-6
        elif source_type=='tt2015' and self.ne == 4: #DC
            v = 0
            w = 0            
            kappa = (m_sol[0] + self.MAXVAL) / 20           ##0, 360
            sigma = m_sol[1] / 40                           ##-90, 90
            h = (m_sol[2] + self.MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[3]+self.MAXVAL)/3600 + 4)   ##Mw: 4-6
        elif source_type=='tt2015' and self.ne == 5: #Deviatoric MT
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
            m_sol[0:6] *= self.M00
            rho,v,w,kappa,sigma,h = to_lune(m_sol[0:6])
        else:
            #source type = Tashiro
            m_sol[:5] = (self.MAXVAL + m_sol[:5]) / 7200   #(0,1)
            m_sol[5] = (self.MAXVAL + m_sol[5]) / 3600 + 4 #Mw 4-6
            mij = Tashiro2MT6(m_sol[:6])
            mij = ned2rtp(mij) #up-south-east convention ##TODO:double check this part
            rho,v,w,kappa,sigma,h = to_lune(mij)
            
        if source_type=='force':
            source_solution = UnstructuredGrid(
                dims=('F0','phi', 'h'),
                coords=(F0, phi, h),
                callback=to_force)   
        else:
            ## moment tensor
            source_solution = UnstructuredGrid(
                dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
                coords=(rho, v,w,kappa,sigma,h),
                callback=to_mt)
            
        ##calculate mean solution of noise and time shifts
        noise_solution = m_sol[self.ne:self.ne+self.ns] * self.noise_scale1 + self.noise_scale2 
        tau_solution = m_sol[self.ne+self.ns: ] * self.time_shift_scale1 + self.time_shift_scale2
        if self.time_shift_groups == 1:
            return source_solution, noise_solution, np.repeat(tau_solution, 2)
        else:
            return source_solution, noise_solution, tau_solution
        
    def save_chains(self, sampler, file_path='./', thin=1):
        flat_samples_out = sampler.get_chain(discard=0, thin=thin, flat=True)

        #transfer the samples into final source parameters, noise and time shifts as did in log likelihood function
        source_type = self.method.split('_')[0] 
        if 'mij' == source_type:
            pass
        elif 'tashiro' == source_type:
            flat_samples_out[:,:self.ne] = np.column_stack((
                                (self.MAXVAL + flat_samples_out[:,:5]) / 7200,   #(0,1)
                                (self.MAXVAL + flat_samples_out[:,5]) / 3600 + 4 #Mw 4-6
                                ))
        elif 'tt2015' == source_type: 
            to_rho_vec = np.vectorize(to_rho)
            if 6 == self.ne:
                flat_samples_out[:,:self.ne] = np.column_stack((
                                flat_samples_out[:, 0] / 10800,
                                flat_samples_out[:, 1] * np.pi / 9600,
                                (flat_samples_out[:, 2] + self.MAXVAL) / 20,
                                flat_samples_out[:, 3] / 40,
                                (flat_samples_out[:, 4] + self.MAXVAL) / 7200,
                                to_rho_vec((flat_samples_out[:, 5] + self.MAXVAL) / 3600 + 4)
                                ))
            elif 5 == self.ne:
                    flat_samples_out[:,:self.ne] = np.column_stack((
                                flat_samples_out[:, 0] / 10800, #w=0
                                (flat_samples_out[:, 1] + self.MAXVAL) / 20,
                                flat_samples_out[:, 2] / 40,
                                (flat_samples_out[:, 3] + self.MAXVAL) / 7200,
                                # to_rho((flat_samples_out[:, 4] + self.MAXVAL) / 3600 + 4)
                                to_rho_vec((flat_samples_out[:, 4] + self.MAXVAL) / 3600 + 4)
                                ))
            elif 4 == self.ne:
                    flat_samples_out[:,:self.ne] = np.column_stack((
                                #v,w=0
                                (flat_samples_out[:, 0] + self.MAXVAL) / 20,
                                flat_samples_out[:, 1] / 40,
                                (flat_samples_out[:, 2] + self.MAXVAL) / 7200,
                                # to_rho((flat_samples_out[:, 3] + self.MAXVAL) / 3600 + 4)
                                to_rho_vec((flat_samples_out[:, 3] + self.MAXVAL) / 3600 + 4)
                                ))
        elif 'force' == source_type:
            flat_samples_out[:,:self.ne] = np.column_stack((
                                (flat_samples_out[:,0]+ self.MAXVAL) / 20,     #[0, 360]
                                flat_samples_out[:,1] / 3600,                  #[-1,1]
                                (flat_samples_out[:,2] + self.MAXVAL) 
                                ))
        elif 'mtsf' == source_type:
            #pass mij
            #force
            flat_samples_out[:,6:self.ne] = np.column_stack((
                                (flat_samples_out[:,6]+ self.MAXVAL) / 20,     #[0, 360]
                                flat_samples_out[:,7] / 3600,                  #[-1,1]
                                (flat_samples_out[:,8] + self.MAXVAL) 
                                ))
        else:
            raise ValueError(f"Unknown method: {source_type}")  
        
        ##tansform the noise and time shift parameters
        flat_samples_out[:,self.ne:self.ne+self.ns] = flat_samples_out[:,self.ne:self.ne+self.ns] * self.noise_scale1 + self.noise_scale2 
        flat_samples_out[:,self.ne+self.ns:] = flat_samples_out[:,self.ne+self.ns:] * self.time_shift_scale1 + self.time_shift_scale2

        #save sampled parameters: source ,noise and time shifts
        self.chain_fname = file_path +'MCMC_sampling_%s_model.npy' % self.method
        np.save(self.chain_fname, flat_samples_out)
        #save log likelihood values
        self.logprob_fname = file_path +'MCMC_sampling_%s_log_prob.npy' % self.method
        log_prob_samples = sampler.get_log_prob(discard=0, thin=thin, flat=True)
        np.save(self.logprob_fname, log_prob_samples)

    def diagnose(self, sampler):
        ## Get the idea of autocorrelation in the sampled chains
        tau = sampler.get_autocorr_time(tol=0)
        print ('\nAutocorrelation time for each coordinates of the model space:\n    ', tau)


 
