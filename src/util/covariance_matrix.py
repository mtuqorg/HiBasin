import os 
import matplotlib
import copy
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
        '''noise_model: uncorrelated, exponential, empirical'''
        #parameter 'data_noise' has the mtuq Dataset
        self.noise_model = noise_model
        self.npts_acf_lag = npts_acf_lag

        ##process the noise using the same way as the data_sw
        for traces in data_noise:
            ## get the pre-event noise data by triming the data based on the origin time
            traces.trim(origin.time - noise_length, origin.time) 
            # traces.trim(origin.time +1000, origin.time +1000 + noise_length) #Test for DPRK2016 tests because the pre-event data is noisy. 

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
            # traces.resample(1)
            tags = traces.tags
            if 'type:velocity' in tags:
                print("Converting velocity to displacement")
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
                noise_std[s,c] = np.std(self.data[s,c],ddof=0)
        return noise_std
    
    def get_acf(self):
        acf = np.zeros((self.ns, self.nc, self.nt))
        for s in range(self.ns):
            for c in range(self.nc):
                acf[s,c] = self._get_acf(self.data[s,c])
        return acf
    
    def calc_exponential_cd(self, scale=10):
        '''
        Generate the covariance matrix for exponential decay noise model
        '''
        x = np.arange(self.npts_acf_lag)
        cov_matrix = np.exp(-np.abs(x[:, None] - x[None, :]) / scale)
        return cov_matrix
    
    def calc_empirical_cd(self, acf):
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
                    cov_d[s, c] = self.calc_exponential_cd(re[0])
            
            return cov_d
        elif self.noise_model == 'empirical':
            ## Calculate the covariance matrix for empirical noise model
            acf = self.get_acf()
            for s in range(self.ns):
                for c in range(self.nc):
                    cov_d[s, c] = self.calc_empirical_cd(acf[s, c, :self.npts_acf_lag])
            np.save('covd_emp', cov_d)
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

    def plot_noise_series(self):
        time_ax = np.arange(self.nt) * self.dt
        fig, axs = plt.subplots(self.nc, self.ns, sharex= True, sharey = True, figsize = (10,4))
        for ist in range(self.ns):
            for ic in range(self.nc):
                axs[ic,ist].plot(time_ax, self.data[ist,ic], lw=0.5) #ns . nc . nt
                axs[0,ist].set_title(self.stations[ist].network + '.' + self.stations[ist].station,fontsize=9)
                axs[-1,ist].set_xlabel('Time (s)')
                axs[ic,0].set_ylabel(self.components[ic])
        plt.tight_layout()
        plt.savefig('noise_series.png', dpi=300)
        plt.close()

    def plot_auto_corr_func(self):
        acf = self.get_acf()[:,:,:self.npts_acf_lag]
        time_ax = np.arange(self.npts_acf_lag) * self.dt

        fig,axes = plt.subplots(3,1, sharex=True, figsize=(7,5))
        for ist in range(self.ns):
            for ic in range(self.nc):
                axes[ic].plot( time_ax, acf[ist,ic] )
                axes[ic].set_ylim([-1,1])
                axes[ic].set_xlim([min(time_ax),max(time_ax)])
                
                axes[ic].text(10,0.75,self.components[ic])

            axes[2].set_xlabel('Lag (samples)',fontsize = 12)
            axes[1].set_ylabel('Autocorrelation', fontsize = 12)
            axes[2].legend([s.network + '.' + s.station for s in self.stations],loc = 'lower right', ncol=3, fontsize = 9)

        #plot the zeros
        for i in range(3):
            axes[i].plot(time_ax, np.zeros(self.npts_acf_lag),'--', color = 'gray', linewidth = 1)
        plt.savefig('acf.png', dpi = 300, bbox_inches = 'tight')

    def plot_data_covariance_matrix(self, figname, sigma_in=None):
        covd = self.get_covariance_matrix()
        ns,nc,nt,_ = covd.shape

        if sigma_in is not None:        
            sigma = sigma_in**2 * 1.0e12
            vmin = np.min(sigma) 
            vmax = np.max(sigma)
        else:
            sigma = np.ones((ns,nc))
            vmin = -1
            vmax = 1
    
        ##plot the covariance matrix for all components of all stations
        fig,axes = plt.subplots(nc,ns, sharex=True, sharey=True, figsize=(9,2.5), subplot_kw={'xticks': [0,150,300], 'yticks': [0,150,300]})
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        # cm = copy.copy( plt.get_cmap('copper').reversed())
        cm = plt.get_cmap('jet')
        for ist in range(ns):
            for ic in range(nc):
                cov_i = covd[ist,ic] * sigma[ist,ic] 
                im = axes[ic,ist].imshow(cov_i, vmin=vmin, vmax=vmax, cmap =cm)
                if ist == 0 and ic == 1:
                    axes[ic,ist].set_ylabel('Time (s)')
                if ist == int(ns/2):
                    axes[ic,ist].set_xlabel('Time (s)')

                axes[0,ist].set_title(self.stations[ist].network + '.' + self.stations[ist].station,fontsize=9)
                axes[ic,ist].set_xlim([0,nt])
                axes[ic,ist].set_ylim([0,nt])
                plt.gca().invert_yaxis()
           
        axes[0,0].annotate('Z', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
        axes[1,0].annotate('R', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
        axes[2,0].annotate('T', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
            
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.81, 0.1, 0.02, 0.8])
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=norm)
        if sigma_in is not None:  
            cb.set_label(label='Covariance amplitude ($10^{12}$)',fontsize=10)
        else:
            cb.set_label(label='Covariance amplitude',fontsize=10)
        #plt.colorbar(im, cax=cax, ax = axes[-1,-1])
        plt.savefig(figname, dpi = 300, bbox_inches = 'tight')



