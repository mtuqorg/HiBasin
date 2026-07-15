import numpy as np
from scipy.linalg import toeplitz, cholesky, solve_triangular
from scipy.optimize import curve_fit
from hibasin.util.acf_fit_hypersweep import fit_two_attenuated_cosine_hypersweep
import matplotlib
import matplotlib.pyplot as plt
from mtuq.misfit.waveform import level2


def _exp_func(x, re):
    return np.exp(-re * x)


def _two_atcosine_func(x, b, re1, L1, re2, L2):
    return (b * np.exp(-x / re1) * np.cos(2 * np.pi * x / L1) +
            (1 - b) * np.exp(-x / re2) * np.cos(2 * np.pi * x / L2))


class covariance_matrix_shared_Cd:
    """
    Noise covariance estimator using two SHARED normalised matrices,
    following Mustać & Tkalčić (2017, BSSA, doi:10.1785/0120160379).

    Two component groupings are supported (component_grouping parameter):

    'vertical_horizontal' (default, Mustać & Tkalčić 2017):
      • C_d0 — mean ACF of all Z seismograms
      • C_d1 — mean ACF of all horizontal seismograms
                (N/E preferred before rotation; R/T fallback)

    'rayleigh_love' (Rayleigh / Love wave separation):
      • C_d0 — mean ACF of all Z and R seismograms  (Rayleigh group)
      • C_d1 — mean ACF of all T seismograms         (Love group)

    The same shared matrices are used for every station; only σ_n²
    (from get_noise_std) varies between stations.

    Supported noise models
    ----------------------
    'empirical'             - Toeplitz matrix built from the mean ACF
    'exponential'           - exponential decay fitted to the mean ACF
    'two_attenuated_cosine' - two-attenuated-cosine model fitted to the
                              mean ACF (eq. 4, Mustać & Tkalčić, 2017)

    Output compatibility
    --------------------
    get_noise_std(), calc_InversionDeterminant_cd(), and the plot methods
    share the same signatures as covariance_matrix_Cd.

    Usage
    -----
    noise_est = covariance_matrix_shared_Cd(
        origin, data_noise, npts_acf_lag, process_sw,
        noise_length=1600, noise_model='two_attenuated_cosine',
        component_grouping='rayleigh_love')

    noise_std_sw          = noise_est.get_noise_std()
    cov_inv, log_cov_det  = noise_est.calc_InversionDeterminant_cd()
    # cov_inv shape: (2, nt, nt)  — compact shared form (NOT same as covariance_matrix_Cd)
    # log_cov_det shape: (ns, nc)
    # Pass component_grouping to MCMC_FullMij as well so the solver uses
    # the same C_d0 / C_d1 assignment.
    """

    def __init__(self, origin, data_noise, npts_acf_lag, process_data,
                 noise_length=3600, noise_model='two_attenuated_cosine',
                 offset_t0=0, component_grouping='vertical_horizontal',
                 fit_method='curve_fit'):
        """
        Parameters
        ----------
        origin : mtuq Origin
        data_noise : mtuq Dataset
            Raw waveforms (unprocessed). Pass in the original ZNE orientation
            when possible: N and E component traces are then used to estimate
            C_d1 for 'vertical_horizontal' grouping.  If only ZRT data are
            available, R and T are used as a fallback.
        npts_acf_lag : int
            Number of time samples in the signal window (= nt of data_sw).
        process_data : mtuq ProcessData
            Used to extract filter settings so that noise and signal are
            processed identically.
        noise_length : float
            Length of the pre-event noise window in seconds.
        noise_model : str
            'empirical', 'exponential', or 'two_attenuated_cosine'.
        offset_t0 : float
            Time offset for the pre-event noise window in seconds.
        component_grouping : str
            'vertical_horizontal' - C_d0 for Z, C_d1 for R/T.
            'rayleigh_love'       - C_d0 for Z+R, C_d1 for T (default).
        """
        if component_grouping not in ('vertical_horizontal', 'rayleigh_love'):
            raise ValueError(
                f"component_grouping '{component_grouping}' is not supported. "
                "Use 'vertical_horizontal' or 'rayleigh_love'.")
        if fit_method not in ('curve_fit', 'hypersweep'):
            raise ValueError(
                f"fit_method '{fit_method}' is not supported. "
                "Use 'curve_fit' or 'hypersweep'.")
        self.component_grouping = component_grouping
        self.noise_model        = noise_model
        self.fit_method         = fit_method
        self.npts_acf_lag = npts_acf_lag
        self.nt, self.dt  = level2._get_time_sampling(data_noise)

        filter_type = process_data.filter_type.lower()
        freq_min    = getattr(process_data, 'freq_min', None)
        freq_max    = getattr(process_data, 'freq_max', None)
        freq        = getattr(process_data, 'freq', None)

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
                raise ValueError(f"Unsupported filter_type: '{process_data.filter_type}'")

            tags = traces.tags
            if 'type:velocity' in tags:
                for trace in traces:
                    trace.data = np.cumsum(trace.data) * self.dt
                tags[tags.index('type:velocity')] = 'type:displacement'

        self.stations   = level2._get_stations(data_noise)
        self.components = level2._get_components(data_noise)
        self.data       = level2._get_data(data_noise, self.stations, self.components)
        self.ns, self.nc, self.nt = self.data.shape

        # Capture N/E traces for C_dh (Mustać & Tkalčić, 2017).
        # level2._get_data only extracts Z/R/T; collect N/E directly here
        # before they are discarded.
        horiz_en          = []
        horiz_en_stations = []
        for traces in data_noise:
            for comp_code in ('N', 'E', '1', '2'):
                sel = traces.select(component=comp_code)
                if sel:
                    seg     = sel[0].data
                    n_avail = len(seg)
                    if n_avail >= self.nt:
                        horiz_en.append(seg[:self.nt].copy())
                    else:
                        tmp = np.zeros(self.nt)
                        tmp[:n_avail] = seg
                        horiz_en.append(tmp)
                    horiz_en_stations.append(sel[0].stats.station)
        self._horiz_en          = np.array(horiz_en) if horiz_en else None
        self._horiz_en_stations = horiz_en_stations

    # ------------------------------------------------------------------
    # ACF helpers
    # ------------------------------------------------------------------

    def _get_acf(self, data_1d):
        acf  = np.correlate(data_1d, data_1d, mode='full')
        half = acf.size // 2
        acf  = acf[half:half + self.nt]
        acf /= acf[0]
        return acf

    def get_acf(self):
        acf = np.zeros((self.ns, self.nc, self.nt))
        for s in range(self.ns):
            for c in range(self.nc):
                acf[s, c] = self._get_acf(self.data[s, c])
        return acf

    def get_noise_std(self):
        noise_std = np.ones((self.ns, self.nc))
        for s in range(self.ns):
            for c in range(self.nc):
                noise_std[s, c] = np.std(self.data[s, c], ddof=0)
        return noise_std

    # ------------------------------------------------------------------
    # Mean ACF for each polarisation group
    # ------------------------------------------------------------------

    def _mean_acf_z(self):
        """Mean ACF (length nt) across all Z seismograms."""
        acf   = self.get_acf()
        z_idx = [i for i, c in enumerate(self.components) if c == 'Z']
        if not z_idx:
            raise ValueError("No Z component found in noise data.")
        return acf[:, z_idx, :].mean(axis=(0, 1))

    def _mean_acf_h(self):
        """
        Mean ACF (length nt) across all horizontal seismograms.
        Uses N/E traces when available (before rotation to ZRT);
        falls back to R/T otherwise.
        """
        if self._horiz_en is not None:
            acf_en = np.array([self._get_acf(seg) for seg in self._horiz_en])
            return acf_en.mean(axis=0)

        acf   = self.get_acf()
        h_idx = [i for i, c in enumerate(self.components)
                 if c in ('R', 'T', 'E', 'N')]
        if not h_idx:
            raise ValueError(
                "No horizontal component (R/T/E/N) found in noise data. "
                "Pass pre-event noise that contains at least one horizontal component."
            )
        return acf[:, h_idx, :].mean(axis=(0, 1))

    def _mean_acf_rz(self):
        """Mean ACF across all Z and R seismograms (Rayleigh wave group)."""
        acf    = self.get_acf()
        rz_idx = [i for i, c in enumerate(self.components) if c in ('Z', 'R')]
        if not rz_idx:
            raise ValueError("No Z or R component found in noise data.")
        return acf[:, rz_idx, :].mean(axis=(0, 1))

    def _mean_acf_t(self):
        """Mean ACF across all T seismograms (Love wave group)."""
        acf   = self.get_acf()
        t_idx = [i for i, c in enumerate(self.components) if c == 'T']
        if not t_idx:
            raise ValueError("No T component found in noise data.")
        return acf[:, t_idx, :].mean(axis=(0, 1))

    # ------------------------------------------------------------------
    # Matrix builders (private)
    # ------------------------------------------------------------------

    def _build_empirical(self, acf_mean):
        return toeplitz(acf_mean[:self.npts_acf_lag])

    def _build_exponential(self, acf_mean):
        lags = np.arange(self.nt) * self.dt
        re, _ = curve_fit(_exp_func, lags, acf_mean, p0=[0.1], maxfev=5000)
        x = np.arange(self.npts_acf_lag) * self.dt
        abs_lag = np.abs(x[:, None] - x[None, :])
        return np.exp(-abs_lag * re[0])

    def _fit_two_attenuated_cosine(self, acf_mean, label=''):
        lags   = np.arange(self.npts_acf_lag) * self.dt
        target = acf_mean[:self.npts_acf_lag]
        T_sig  = self.npts_acf_lag * self.dt

        if self.fit_method == 'hypersweep':
            return fit_two_attenuated_cosine_hypersweep(
                lags, target, T_sig=T_sig)   # signal window length
        # Cap re1/re2 so that exp(-T_sig/re) <= exp(-3) ~ 0.05 at the
        # end of the lag window, forcing the ACF model to decay to near-zero
        # within the signal window (Mustać & Tkalčić 2016/2017).
        re_max = T_sig / 3.0
        bounds = ([0, 1e-6, 1e-6, 1e-6, 1e-6],
                  [1, re_max, np.inf, re_max, np.inf])

        # Try several starting points; keep the lowest-residual solution.
        # p0 re values are scaled by T_sig to stay within bounds.
        p0_list = [
            [0.5, T_sig*0.10, T_sig*0.05, T_sig*0.10, T_sig*0.10],
            [0.5, T_sig*0.25, T_sig*0.05, T_sig*0.05, T_sig*0.15],
            [0.7, T_sig*0.05, T_sig*0.05, T_sig*0.25, T_sig*0.10],
            [0.3, T_sig*0.20, T_sig*0.10, T_sig*0.10, T_sig*0.05],
            [0.5, T_sig*0.30, T_sig*0.05, T_sig*0.10, T_sig*0.20],
        ]

        best_popt = None
        best_res  = np.inf
        for p0 in p0_list:
            try:
                popt, _ = curve_fit(_two_atcosine_func, lags, target,
                                    p0=p0, bounds=bounds, maxfev=10000)
                res = np.sum((target - _two_atcosine_func(lags, *popt)) ** 2)
                if res < best_res:
                    best_res  = res
                    best_popt = popt
            except RuntimeError:
                continue

        if best_popt is None:
            tag = f' ({label})' if label else ''
            print(f'WARNING: two_attenuated_cosine fit failed{tag}. '
                  'All initial guesses diverged. '
                  'Falling back to first p0 — covariance matrix will be approximate. '
                  'Consider noise_model="empirical" or "exponential".')
            best_popt = p0_list[0]

        return tuple(best_popt)

    def _build_two_attenuated_cosine(self, acf_mean):
        b, re1, L1, re2, L2 = self._fit_two_attenuated_cosine(acf_mean)
        lags    = np.arange(self.npts_acf_lag) * self.dt
        abs_lag = np.abs(lags[:, None] - lags[None, :])
        return (b * np.exp(-abs_lag / re1) * np.cos(2 * np.pi * abs_lag / L1) +
                (1 - b) * np.exp(-abs_lag / re2) * np.cos(2 * np.pi * abs_lag / L2))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_shared_covariance_matrices(self):
        """
        Build the two shared normalised covariance matrices.

        Grouping is controlled by self.component_grouping:
          'vertical_horizontal': C_d0 from mean Z ACF, C_d1 from mean H ACF
          'rayleigh_love':       C_d0 from mean Z+R ACF, C_d1 from mean T ACF

        Returns
        -------
        C_d0 : ndarray (npts_acf_lag, npts_acf_lag)  — group-0 matrix
        C_d1 : ndarray (npts_acf_lag, npts_acf_lag)  — group-1 matrix
        """
        if self.component_grouping == 'rayleigh_love':
            acf_0 = self._mean_acf_rz()
            acf_1 = self._mean_acf_t()
        else:
            acf_0 = self._mean_acf_z()
            acf_1 = self._mean_acf_h()

        def _build(acf):
            if self.noise_model == 'empirical':
                return self._build_empirical(acf)
            elif self.noise_model == 'exponential':
                return self._build_exponential(acf)
            elif self.noise_model == 'two_attenuated_cosine':
                return self._build_two_attenuated_cosine(acf)
            else:
                raise ValueError(
                    f"noise_model '{self.noise_model}' is not supported. "
                    "Use 'empirical', 'exponential', or 'two_attenuated_cosine'.")

        return _build(acf_0), _build(acf_1)

    def _is_group0(self, comp):
        """Return True if component belongs to the first covariance group."""
        if self.component_grouping == 'rayleigh_love':
            return comp in ('Z', 'R')
        return comp == 'Z'

    def get_covariance_matrix(self):
        """
        Return cov_d of shape (ns, nc, npts_acf_lag, npts_acf_lag).
        Components in group 0 are filled with C_d0; group 1 with C_d1.
        """
        C_d0, C_d1 = self.get_shared_covariance_matrices()
        cov_d = np.empty((self.ns, self.nc, self.npts_acf_lag, self.npts_acf_lag))
        for c, comp in enumerate(self.components):
            cov_d[:, c, :, :] = C_d0 if self._is_group0(comp) else C_d1
        return cov_d

    def calc_InversionDeterminant_cd(self):
        """
        Invert the two shared covariance matrices with exactly 2 Cholesky
        decompositions.

        Returns a COMPACT form that the _MCMC_Base solver recognises
        automatically (detected by cov_inv.ndim == 3):

        cov_inv     : ndarray (2, nt, nt)   [0] = C_d0^{-1},  [1] = C_d1^{-1}
        log_cov_det : ndarray (ns, nc)       log sqrt|C_shared| per seismogram

        Pass the same component_grouping to MCMC_FullMij so the solver
        assigns C_d0 / C_d1 to the correct component columns.
        """
        C_d0, C_d1 = self.get_shared_covariance_matrices()
        nt = self.npts_acf_lag

        def _factor(C):
            L     = cholesky(C, lower=True)
            ld    = np.sum(np.log(np.abs(np.diag(L))))
            L_inv = solve_triangular(L, np.eye(nt), lower=True)
            return L_inv.T @ L_inv, ld

        C_d0_inv, log_det_0 = _factor(C_d0)
        C_d1_inv, log_det_1 = _factor(C_d1)

        cov_inv = np.stack([C_d0_inv, C_d1_inv])   # (2, nt, nt)

        log_cov_det = np.empty((self.ns, self.nc))
        for c, comp in enumerate(self.components):
            log_cov_det[:, c] = log_det_0 if self._is_group0(comp) else log_det_1

        return cov_inv, log_cov_det

    # ------------------------------------------------------------------
    # Diagnostic plots (same signatures as covariance_matrix_Cd)
    # ------------------------------------------------------------------

    def plot_noise_series(self, figname='noise_series.png'):
        time_ax = np.arange(self.nt) * self.dt
        fig, axs = plt.subplots(self.nc, self.ns, sharex=True, sharey=True,
                                figsize=(10, 4))
        for ist in range(self.ns):
            for ic in range(self.nc):
                axs[ic, ist].plot(time_ax, self.data[ist, ic], lw=0.5)
                axs[0, ist].set_title(
                    self.stations[ist].network + '.' + self.stations[ist].station,
                    fontsize=9)
                axs[-1, ist].set_xlabel('Time (s)')
                axs[ic, 0].set_ylabel(self.components[ic])
        plt.tight_layout()
        plt.savefig(figname, dpi=300)
        plt.close()

    def plot_auto_corr_func(self, figname='acf.png', legend=True):
        acf = self.get_acf()[:, :, :self.npts_acf_lag]
        time_ax = np.arange(self.npts_acf_lag) * self.dt

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 5))
        for ist in range(self.ns):
            for ic in range(self.nc):
                axes[ic].plot(time_ax, acf[ist, ic],alpha=0.5, lw=0.8)
                axes[ic].set_ylim([-1, 1])
                axes[ic].set_xlim([min(time_ax), max(time_ax)])
                axes[ic].text(10, 0.75, self.components[ic])
            axes[2].set_xlabel('Lag (s)', fontsize=12)
            axes[1].set_ylabel('Autocorrelation', fontsize=12)
            if legend:
                axes[2].legend(
                    [s.network + '.' + s.station for s in self.stations],
                    loc='lower right', ncol=3, fontsize=9)
        for i in range(3):
            axes[i].plot(time_ax, np.zeros(self.npts_acf_lag), '--',
                         color='gray', linewidth=1)

        # Overlay the mean ACFs and (optionally) fitted curves
        if self.component_grouping == 'rayleigh_love':
            acf_0_full = self._mean_acf_rz()
            acf_1_full = self._mean_acf_t()
            g0_idx = [i for i, c in enumerate(self.components) if c in ('Z', 'R')]
            g1_idx = [i for i, c in enumerate(self.components) if c == 'T']
            label_0, label_1 = 'mean Z+R (Rayleigh)', 'mean T (Love)'
        else:
            acf_0_full = self._mean_acf_z()
            acf_1_full = self._mean_acf_h()
            g0_idx = [i for i, c in enumerate(self.components) if c == 'Z']
            g1_idx = [i for i, c in enumerate(self.components) if c != 'Z']
            label_0, label_1 = 'shared mean Z', 'shared mean H'

        acf_0 = acf_0_full[:self.npts_acf_lag]
        acf_1 = acf_1_full[:self.npts_acf_lag]

        for i in g0_idx:
            axes[i].plot(time_ax, acf_0, 'k-', lw=2, label=label_0)
            axes[i].legend(fontsize=8)
        for i in g1_idx:
            axes[i].plot(time_ax, acf_1, 'k-', lw=2, label=label_1)
        if g1_idx:
            axes[g1_idx[0]].legend(fontsize=8)

        if self.noise_model == 'two_attenuated_cosine':
            popt_0 = self._fit_two_attenuated_cosine(acf_0_full)
            popt_1 = self._fit_two_attenuated_cosine(acf_1_full)
            fit_0  = _two_atcosine_func(time_ax, *popt_0)
            fit_1  = _two_atcosine_func(time_ax, *popt_1)
            for i in g0_idx:
                axes[i].plot(time_ax, fit_0, 'k--', lw=1.5, label='fitted model')
                axes[i].legend(fontsize=8)
            for i in g1_idx:
                axes[i].plot(time_ax, fit_1, 'k--', lw=1.5, label='fitted model')
            if g1_idx:
                axes[g1_idx[0]].legend(fontsize=8)
        elif self.noise_model == 'exponential':
            lags_full = np.arange(self.nt) * self.dt
            for acf_full, idx, label in [
                    (acf_0_full, g0_idx, 'exp fit'),
                    (acf_1_full, g1_idx, 'exp fit')]:
                try:
                    re, _ = curve_fit(_exp_func, lags_full, acf_full,
                                      p0=[0.1], maxfev=5000)
                    fit = _exp_func(time_ax, re[0])
                except RuntimeError:
                    fit = _exp_func(time_ax, 0.1)
                for i in idx:
                    axes[i].plot(time_ax, fit, 'k--', lw=1.5, label=label)
                    axes[i].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_data_covariance_matrix(self, figname, sigma_in=None):
        covd = self.get_covariance_matrix()
        ns, nc, nt, _ = covd.shape

        if sigma_in is not None:
            sigma = sigma_in ** 2 * 1.0e12
            vmin  = np.min(sigma)
            vmax  = np.max(sigma)
        else:
            sigma = np.ones((ns, nc))
            vmin  = np.min(covd)
            vmax  = np.max(covd)

        fig_width = 1.5 * ns
        fig, axes = plt.subplots(nc, ns, sharex=True, sharey=True,
                                 figsize=(fig_width, 2.5),
                                 subplot_kw={'xticks': [0, 150, 300],
                                             'yticks': [0, 150, 300]})
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cm   = plt.get_cmap('jet')
        for ist in range(ns):
            for ic in range(nc):
                cov_i = covd[ist, ic] * sigma[ist, ic]
                axes[ic, ist].imshow(cov_i, vmin=vmin, vmax=vmax, cmap=cm)
                if ist == 0 and ic == 1:
                    axes[ic, ist].set_ylabel('Time (s)')
                if ist == int(ns / 2):
                    axes[ic, ist].set_xlabel('Time (s)')
                axes[0, ist].set_title(
                    self.stations[ist].network + '.' + self.stations[ist].station,
                    fontsize=9)
                axes[ic, ist].set_xlim([0, nt])
                axes[ic, ist].set_ylim([0, nt])
                plt.gca().invert_yaxis()

        axes[0, 0].annotate('Z', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
        axes[1, 0].annotate('R', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
        axes[2, 0].annotate('T', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.81, 0.1, 0.02, 0.8])
        cb  = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=norm)
        cb.set_label(
            label='Covariance amplitude ($10^{12}$)' if sigma_in is not None
                  else 'Covariance amplitude',
            fontsize=10)
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
