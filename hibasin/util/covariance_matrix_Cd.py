import os 
import matplotlib
import copy
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.linalg import cholesky, solve_triangular
from mtuq.misfit.waveform import level2 
from hibasin.util.math import exponential_covariance, calc_InversionDeterminant_cd

#exponential decay function
def exp_func(x, re):
    return np.exp(-re*x)

# two attenuated cosine ACF model (Mustać and Tkalčić, 2016, GJI)
# C(tau) = b*exp(-tau/re1)*cos(2pi*tau/L1) + (1-b)*exp(-tau/re2)*cos(2pi*tau/L2)
def two_atcosine_func(x, b, re1, L1, re2, L2):
    return (b * np.exp(-x / re1) * np.cos(2*np.pi*x / L1) +
            (1-b) * np.exp(-x / re2) * np.cos(2*np.pi*x / L2))

class covariance_matrix_Cd:
    def __init__(self, origin, data_noise, npts_acf_lag, process_data,
                 noise_length=3600, noise_model='uncorrelated', offset_t0=0):
        """
        Parameters
        ----------
        origin : mtuq Origin
        data_noise : mtuq Dataset
            Raw waveforms (unprocessed). The pre-event window is extracted
            and filtered here using the same settings as process_data.
        npts_acf_lag : int
            Number of time samples in the signal window (= nt of data_sw).
        process_data : mtuq ProcessData
            The same ProcessData object used to process the observed data.
            Filter type and frequencies are read from it directly, guaranteeing
            identical processing for noise and signal.
        noise_length : float
            Length of the pre-event noise window in seconds.
        noise_model : str
            'uncorrelated', 'exponential', 'empirical', or 'two_attenuated_cosine'.
        offset_t0 : float
            Time offset for the pre-event noise window in seconds.
        """
        self.noise_model = noise_model
        self.npts_acf_lag = npts_acf_lag
        self.nt, self.dt = level2._get_time_sampling(data_noise)

        # Extract filter settings from process_data so processing is identical
        # to the observed data
        filter_type = process_data.filter_type.lower()
        freq_min = getattr(process_data, 'freq_min', None)
        freq_max = getattr(process_data, 'freq_max', None)
        freq     = getattr(process_data, 'freq', None)

        for traces in data_noise:
            traces.trim(origin.time - noise_length - offset_t0, origin.time - offset_t0)

            if filter_type == 'bandpass':
                for trace in traces:
                    trace.detrend('demean')
                    trace.detrend('linear')
                    trace.taper(0.05, type='hann')
                    trace.filter('bandpass', zerophase=False,
                                 freqmin=freq_min, freqmax=freq_max)

            elif filter_type == 'lowpass':
                for trace in traces:
                    trace.detrend('demean')
                    trace.detrend('linear')
                    trace.taper(0.05, type='hann')
                    trace.filter('lowpass', zerophase=False, freq=freq)

            elif filter_type == 'highpass':
                for trace in traces:
                    trace.detrend('demean')
                    trace.detrend('linear')
                    trace.taper(0.05, type='hann')
                    trace.filter('highpass', zerophase=False, freq=freq)

            else:
                raise ValueError(f"Unsupported filter_type in process_data: "
                                 f"'{process_data.filter_type}'")

            tags = traces.tags
            if 'type:velocity' in tags:
                for trace in traces:
                    trace.data = np.cumsum(trace.data) * self.dt
                tags[tags.index('type:velocity')] = 'type:displacement'

        # collect metadata
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
        x = np.arange(self.npts_acf_lag) * self.dt
        cov_matrix = np.exp(-np.abs(x[:, None] - x[None, :]) * scale)
        return cov_matrix
    
    def calc_empirical_cd(self, acf):
        '''
        Generate the covariance matrix for empirical noise model
        '''
        return toeplitz(acf, acf)

    def calc_two_attenuated_cosine_cd(self, b, re1, L1, re2, L2):
        '''
        Generate the covariance matrix for two attenuated cosine noise model.
        Parameters are in time units (seconds): re1, re2 are decay lengths, L1, L2 are periods.
        '''
        lags = np.arange(self.npts_acf_lag) * self.dt  # lag axis in seconds
        abs_lag = np.abs(lags[:, None] - lags[None, :])
        cov_matrix = (b * np.exp(-abs_lag / re1) * np.cos(2*np.pi*abs_lag / L1) +
                      (1-b) * np.exp(-abs_lag / re2) * np.cos(2*np.pi*abs_lag / L2))
        return cov_matrix

    def _fit_two_attenuated_cosine(self, acf_1d, label=''):
        """
        Multi-start curve_fit for the two-attenuated-cosine ACF model.

        Fits over the first npts_acf_lag samples.  Tries five starting points
        and keeps the solution with the lowest residual SSE.  Consistent with
        covariance_matrix_shared_Cd._fit_two_attenuated_cosine.

        Returns (b, re1, L1, re2, L2).
        """
        lags  = np.arange(self.npts_acf_lag) * self.dt
        target = acf_1d[:self.npts_acf_lag]
        T_sig  = self.npts_acf_lag * self.dt
        re_max = T_sig / 3.0
        bounds = ([0, 1e-6, 1e-6, 1e-6, 1e-6],
                  [1, re_max, np.inf, re_max, np.inf])
        p0_list = [
            [0.5, T_sig * 0.10, T_sig * 0.05, T_sig * 0.10, T_sig * 0.10],
            [0.5, T_sig * 0.25, T_sig * 0.05, T_sig * 0.05, T_sig * 0.15],
            [0.7, T_sig * 0.05, T_sig * 0.05, T_sig * 0.25, T_sig * 0.10],
            [0.3, T_sig * 0.20, T_sig * 0.10, T_sig * 0.10, T_sig * 0.05],
            [0.5, T_sig * 0.30, T_sig * 0.05, T_sig * 0.10, T_sig * 0.20],
        ]
        best_popt = None
        best_res  = np.inf
        for p0 in p0_list:
            try:
                popt, _ = curve_fit(two_atcosine_func, lags, target,
                                    p0=p0, bounds=bounds, maxfev=10000)
                res = np.sum((target - two_atcosine_func(lags, *popt)) ** 2)
                if res < best_res:
                    best_res  = res
                    best_popt = popt
            except RuntimeError:
                continue
        if best_popt is None:
            tag = f' ({label})' if label else ''
            print(f'WARNING: two_attenuated_cosine fit failed{tag}. '
                  'All initial guesses diverged. '
                  'Falling back to first p0 — covariance matrix will be approximate.')
            best_popt = p0_list[0]
        return tuple(best_popt)

    def get_covariance_matrix(self):
        cov_d = np.empty((self.ns, self.nc, self.npts_acf_lag, self.npts_acf_lag))

        if self.noise_model == 'exponential':
            acf  = self.get_acf()
            lags = np.arange(self.npts_acf_lag) * self.dt
            for s in range(self.ns):
                for c in range(self.nc):
                    try:
                        re, _ = curve_fit(exp_func, lags, acf[s, c, :self.npts_acf_lag],
                                          p0=[0.1], maxfev=5000)
                    except RuntimeError:
                        re = [0.1]
                    cov_d[s, c] = self.calc_exponential_cd(re[0])
            
            return cov_d
        elif self.noise_model == 'empirical':
            ## Calculate the covariance matrix for empirical noise model
            acf = self.get_acf()
            for s in range(self.ns):
                for c in range(self.nc):
                    cov_d[s, c] = self.calc_empirical_cd(acf[s, c, :self.npts_acf_lag])
          
            return cov_d
        elif self.noise_model == 'two_attenuated_cosine':
            acf = self.get_acf()
            for s in range(self.ns):
                for c in range(self.nc):
                    label = f'{self.stations[s].network}.{self.stations[s].station} {self.components[c]}'
                    b, re1, L1, re2, L2 = self._fit_two_attenuated_cosine(acf[s, c], label)
                    cov_d[s, c] = self.calc_two_attenuated_cosine_cd(b, re1, L1, re2, L2)

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
                # covL /= np.exp(factor / nt)

                # Invert combined matrix
                covL_inv = solve_triangular(covL, np.eye(nt), lower=True)
                cov_inv[ist,ic] = np.matmul(covL_inv.T, covL_inv)
                log_cov_det[ist,ic] = factor
        return cov_inv, log_cov_det

    def plot_noise_series(self, figname='noise_series.png'):
        time_ax = np.arange(self.nt) * self.dt
        fig, axs = plt.subplots(self.nc, self.ns, sharex=True, sharey=True, figsize=(10, 4))
        for ist in range(self.ns):
            for ic in range(self.nc):
                axs[ic, ist].plot(time_ax, self.data[ist, ic], lw=0.5)
                axs[0, ist].set_title(self.stations[ist].network + '.' + self.stations[ist].station, fontsize=9)
                axs[-1, ist].set_xlabel('Time (s)')
                axs[ic, 0].set_ylabel(self.components[ic])
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.close()

    def plot_auto_corr_func(self, figname='acf.png'):
        acf_full = self.get_acf()                            # (ns, nc, nt)
        acf      = acf_full[:, :, :self.npts_acf_lag]
        time_ax  = np.arange(self.npts_acf_lag) * self.dt

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 5))
        for ist in range(self.ns):
            for ic in range(self.nc):
                axes[ic].plot(time_ax, acf[ist, ic], alpha=0.5, lw=0.8)
                axes[ic].set_ylim([-1, 1])
                axes[ic].set_xlim([min(time_ax), max(time_ax)])
                axes[ic].text(10, 0.75, self.components[ic])
            axes[2].set_xlabel('Lag (s)', fontsize=12)
            axes[1].set_ylabel('Autocorrelation', fontsize=12)
            axes[2].legend([s.network + '.' + s.station for s in self.stations],
                           loc='lower right', ncol=3, fontsize=9)

        for i in range(3):
            axes[i].plot(time_ax, np.zeros(self.npts_acf_lag), '--', color='gray', linewidth=1)

        if self.noise_model == 'two_attenuated_cosine':
            for ist in range(self.ns):
                for ic in range(self.nc):
                    popt = self._fit_two_attenuated_cosine(acf_full[ist, ic])
                    axes[ic].plot(time_ax, two_atcosine_func(time_ax, *popt),
                                  'k--', lw=1.2)
        elif self.noise_model == 'exponential':
            time_full = np.arange(self.nt) * self.dt
            for ist in range(self.ns):
                for ic in range(self.nc):
                    try:
                        re, _ = curve_fit(exp_func, time_full, acf_full[ist, ic],
                                          p0=[0.1], maxfev=5000)
                    except RuntimeError:
                        re = [0.1]
                    axes[ic].plot(time_ax, exp_func(time_ax, re[0]), 'k--', lw=1.2)

        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()

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
        fig_width = 1.5 * ns
        fig,axes = plt.subplots(nc,ns, sharex=True, sharey=True, figsize=(fig_width,2.5), subplot_kw={'xticks': [0,150,300], 'yticks': [0,150,300]})
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
        plt.close()



