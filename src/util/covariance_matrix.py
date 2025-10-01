import os 
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import cholesky, solve_triangular
from mtuq.misfit.waveform import level2 
from src.util.math import exponential_covariance, calc_InversionDeterminant_cd

#exponential decay function
def exp_func(x, re):
    return np.exp(-re*x)

class covariace_matrix:
    def __init__(self, origin, data_noise, npts_acf_lag, noise_length=3600, filter_type='bandpass', freq_min=0.02, freq_max=0.05, freq=0.05, noise_model='uncorrelated'):
        #parameter 'data_noise' has the mtuq Dataset
        self.noise_model = noise_model
        self.npts_acf_lag = npts_acf_lag

        ##process the noise using the same way as the data_sw
        for traces in data_noise:
            ## get the pre-event noise data by triming the data based on the origin time
            traces.trim(origin.time - noise_length, origin.time) 

            #copy from Processdata in mtuq to make the noise and signal to be processed in the same way
            if filter_type == 'bandpass':
                for trace in traces:
                    trace.detrend('demean')
                    trace.detrend('linear')
                    trace.taper(0.05, type='hann')
                    trace.filter('bandpass', zerophase=False,
                                freqmin=freq_min,
                                freqmax=freq_max)
                    # print("Bandpass filter applied between %.2f - %.2f Hz" % (freq_min, freq_max))

            elif filter_type == 'lowpass':
                for trace in traces:
                    trace.detrend('demean')
                    trace.detrend('linear')
                    trace.taper(0.05, type='hann')
                    trace.filter('lowpass', zerophase=False,
                                freq=freq)

            elif filter_type == 'highpass':
                for trace in traces:
                    trace.detrend('demean')
                    trace.detrend('linear')
                    trace.taper(0.05, type='hann')
                    trace.filter('highpass', zerophase=False,
                                freq=freq)

            tags = traces.tags
            if 'type:velocity' in tags:
                # convert to displacement
                for trace in traces:
                    trace.data = np.cumsum(trace.data)*self.dt
                index = tags.index('type:velocity')
                tags[index] = 'type:displacement'

        # collect metadata
        self.nt, self.dt = level2._get_time_sampling(data_noise)
        self.stations = level2._get_stations(data_noise)
        self.components = level2._get_components(data_noise)

        #
        # collapse main structures into NumPy arrays
        #
        self.data = level2._get_data(data_noise, self.stations, self.components) 
        self.ns, self.nc, self.nt = self.data.shape

    def _get_acf(self, data_1d):
        acf = np.correlate(data_1d, data_1d, mode='full')
        half = acf.size // 2
        acf = acf[half:half + self.nt] 
        acf /= acf[0] #normalized to 1
        return acf
    
    def get_noise_std(self):
        #calculate the pre-event noise strength measured by the rms
        noise_std = np.ones((self.ns, self.nc)) 
        for s in range(self.ns):
            for c in range(self.nc):
                noise_std[s,c] = np.std(self.data[s,c])
        return noise_std
    
    def get_acf(self):
        acf = np.zeros((self.ns, self.nc, self.nt))
        for s in range(self.ns):
            for c in range(self.nc):
                acf[s,c] = self._get_acf(self.data[s,c])
        return acf
    
    def calc_exponential_cd(length, scale=10):
        '''
        Generate the covariance matrix for exponential decay noise model
        '''
        x = np.arange(length)
        cov_matrix = np.exp(-np.abs(x[:, None] - x[None, :]) / scale)
        return cov_matrix
    
    def calc_empirical_cd(acf):
        '''
        Generate the covariance matrix for empirical noise model
        '''
        return toeplitz(acf, acf)

    def get_covariance_matrix(self):
        cov_d = np.empty((self.ns, self.nc, self.npts_acf_lag, self.npts_acf_lag))

        if self.noise_model == 'exponential':
            ## Calculate the covariance matrix for exponential decay noise model
            #acf for all stations and components
            acf = self.get_acf()
            time = np.arange(self.nt) * self.dt
            for s in range(self.ns):
                for c in range(self.nc):
                    re, _ = curve_fit(exp_func, time, acf[s,c])
                    cov_d[s, c] = self.calc_exponential_cd(self.npts_acf_lag, re)

            return cov_d
        elif self.noise_model == 'empirical':
            ## Calculate the covariance matrix for empirical noise model
            acf = self.get_acf()
            for s in range(self.ns):
                for c in range(self.nc):
                    cov_d[s, c] = self.calc_empirical_cd(acf[s, c, :self.npts_acf_lag])
            return cov_d
        else:
            raise ValueError(f"Unknown noise model: {self.noise_model}")
    
    def calc_InversionDeterminant_cd(self):
        '''
        Compute the inverse of matrix N-by-N cov_d, where N is the number of samples
        '''
        cov_d = self.get_covariance_matrix()
        ns,nc,nt,_ = cov_d.shape
        cov_inv = np.zeros((ns,nc,nt,nt))
        log_cov_det = np.zeros((ns,nc))
        # Cholesky decomposition to obtain lower matrix
        for ist in range(ns):
            for ic in range(nc):
                cov = cov_d[ist,ic] 
                covL = cholesky(cov, lower=True)
                #log of sqrt determinant
                factor = np.sum(np.log(np.abs(np.diag(covL))))
                covL /= np.exp(factor / nt)
                
                # Invert combined matrix
                covL_inv = solve_triangular(covL, np.eye(nt), lower=True)
                cov_inv[ist,ic] = np.matmul(covL_inv.T, covL_inv)
                log_cov_det[ist,ic] = factor
        return cov_inv, log_cov_det

    # def plot_noise_series(self):
    #     time_ax = np.arange(self.nt) * self.dt
    #     fig, axs = plt.subplots(self.nc, self.ns, sharex=True, sharey = True, figsize = (10,4))
    #     for ist in range(self.ns):
    #         for ic in range(self.nc):
    #             axs[ic,ist].plot(time_ax, self.data[ist,ic]) #ns . nc . nt
    #             axs[0,ist].set_title(self.stations[ist].split('.')[1],fontsize=9)
    #             axs[-1,ist].set_xlabel('Time (s)')
    #             axs[ic,0].set_ylabel(self.components[ic])
    #     plt.tight_layout()
    #     plt.savefig('noise_series_plots.jpg', dpi=300)
    #     plt.close()




