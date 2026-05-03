import numpy as np
from scipy.linalg import toeplitz, cholesky, solve_triangular, LinAlgError
import matplotlib
import matplotlib.pyplot as plt


class covariance_matrix_empirical_Cd:
    """
    Empirical data covariance matrix estimated from waveform residuals at the
    MAP solution (Mustać et al., 2020, Bull. Seismol. Soc. Am.).

    Each block of C_D is a normalized Toeplitz matrix built from the
    autocorrelation of the residual for that seismogram:

        residual[s, c] = obs[s, c] - pred[s,c])

    This captures both observational noise and theory errors (Green's function
    misfit) simultaneously, making it better suited to nuclear explosion MTI
    than pre-event noise estimates.  The key advantage: because the residuals
    are constrained in time to the signal window, the resulting ACFs have
    short effective correlation lengths and do not suppress the long-period
    isotropic signal the way a pre-event ambient noise ACF does.

    Workflow
    --------
    Phase 1:  MCMC with uncorrelated (diagonal) C_D — recover MT + time shifts
    Between:  Build empirical C_D from Phase-1 MAP residuals  ← this class
    Phase 2:  MCMC with empirical C_D, time shifts fixed

    Usage
    -----
    # After Phase 1:
    mij_map = solver1.get_map_mij(sampler1,
                                   warm_up_steps=int(0.5 * nsteps_1), thin=100)
    emp_cd  = covariance_matrix_empirical_Cd(
                  solver1.obs, solver1.greens, mij_map, npts_acf_lag)
    noise_std_2          = emp_cd.get_noise_std()
    cov_inv, log_cov_det = emp_cd.calc_InversionDeterminant_cd()

    # Phase 2:
    solver2 = MCMC_FullMij(
        misfit_sw, data_sw, greens_sw, noise_std_2,
        cov_inv=cov_inv, log_cov_det=log_cov_det,
        noise_type='correlated', no_time_shift=True)

    Output compatibility
    --------------------
    calc_InversionDeterminant_cd() returns the same (cov_inv, log_cov_det)
    shapes as covariance_matrix_Cd: (ns, nc, nt, nt) and (ns, nc).
    """

    def __init__(self, obs, greens, mij_map, npts_acf_lag):
        """
        Parameters
        ----------
        obs : ndarray (ns, nc, nt)
            Processed observed data — use solver.obs from the Phase-1 solver.
        greens : ndarray (ns, nc, ne, nt)
            Processed, time-shifted Green's functions — use solver.greens from
            the Phase-1 solver (already scaled by M00 for MCMC_FullMij).
        mij_map : ndarray (ne,)
            MAP moment-tensor coefficients in the solver's internal units.
            Obtain via solver1.get_map_mij(sampler1, warm_up_steps, thin).
        npts_acf_lag : int
            Signal window length in samples (= nt of data_sw).
        """
        ns, nc, nt = obs.shape
        self.ns           = ns
        self.nc           = nc
        self.npts_acf_lag = npts_acf_lag

        pred      = np.einsum('scet,e->sct', greens, mij_map)  # (ns, nc, nt)
        residuals = obs - pred                                  # (ns, nc, nt)

        self._obs       = obs
        self._pred      = pred
        self._residuals = residuals

        n = npts_acf_lag
        self._acf       = np.zeros((ns, nc, n))
        self._noise_std = np.zeros((ns, nc))

        for s in range(ns):
            for c in range(nc):
                r = residuals[s, c, :nt]
                self._noise_std[s, c] = np.std(r, ddof=0)

                acf  = np.correlate(r, r, mode='full')
                half = acf.size // 2
                acf  = acf[half:half + n]
                if acf[0] > 0:
                    acf = acf / acf[0]
                else:
                    acf    = np.zeros(n)
                    acf[0] = 1.0
                self._acf[s, c] = acf

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_noise_std(self):
        """
        Return the RMS amplitude of the Phase-1 residuals per seismogram.
        Shape: (ns, nc).  Use as noise_std_sw for the Phase-2 solver.
        """
        return self._noise_std.copy()

    def calc_InversionDeterminant_cd(self):
        """
        Invert each per-seismogram Toeplitz covariance matrix via Cholesky.

        Returns
        -------
        cov_inv     : ndarray (ns, nc, npts_acf_lag, npts_acf_lag)
        log_cov_det : ndarray (ns, nc)   — log of sqrt(det(C)) per seismogram
        """
        n           = self.npts_acf_lag
        cov_inv     = np.zeros((self.ns, self.nc, n, n))
        log_cov_det = np.zeros((self.ns, self.nc))

        for s in range(self.ns):
            for c in range(self.nc):
                C = toeplitz(self._acf[s, c])
                try:
                    L = cholesky(C, lower=True)
                except LinAlgError:
                    # ACF Toeplitz not PD (numerical edge case): add small nugget
                    nugget = 1e-6 * np.eye(n)
                    L = cholesky(C + nugget, lower=True)
                ld                = np.sum(np.log(np.diag(L)))
                L_inv             = solve_triangular(L, np.eye(n), lower=True)
                cov_inv[s, c]     = L_inv.T @ L_inv
                log_cov_det[s, c] = ld

        return cov_inv, log_cov_det

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def plot_residual_acf(self, figname='acf_empirical_Cd.png',
                          component_labels=('Z', 'R', 'T'),
                          station_labels=None):
        """
        Plot per-station residual ACFs used to build C_D.
        Thick black line = ACF for each seismogram.
        """
        n        = self.npts_acf_lag
        time_ax  = np.arange(n)

        fig, axes = plt.subplots(self.nc, 1, sharex=True, figsize=(7, 5))
        if self.nc == 1:
            axes = [axes]

        for s in range(self.ns):
            lbl = (station_labels[s] if station_labels is not None
                   else f'sta{s}')
            for c in range(self.nc):
                axes[c].plot(time_ax, self._acf[s, c], lw=0.8, alpha=0.6,
                             label=lbl)

        for c in range(self.nc):
            axes[c].axhline(0, color='gray', lw=0.8, ls='--')
            axes[c].set_ylim([-1, 1])
            comp = component_labels[c] if c < len(component_labels) else str(c)
            axes[c].text(0.02, 0.85, comp, transform=axes[c].transAxes,
                         fontsize=10)

        axes[-1].set_xlabel('Lag (samples)')
        axes[self.nc // 2].set_ylabel('Normalized ACF')
        axes[0].legend(loc='upper right', ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_data_covariance_matrix(self, figname, sigma_in=None,
                                    component_labels=('Z', 'R', 'T'),
                                    station_labels=None):
        """
        Plot the per-seismogram empirical covariance matrices.

        Parameters
        ----------
        figname : str
        sigma_in : ndarray (ns, nc), optional
            Per-seismogram noise amplitudes.  If provided the matrix is scaled
            by sigma_in**2 * 1e12 and the colour bar is labelled accordingly.
        component_labels : sequence of str
        station_labels : sequence of str, optional
        """
        n  = self.npts_acf_lag
        ns = self.ns
        nc = self.nc

        # Build Toeplitz matrices from stored ACFs
        covd = np.array([[toeplitz(self._acf[s, c]) for c in range(nc)]
                         for s in range(ns)])   # (ns, nc, n, n)

        if sigma_in is not None:
            sigma = sigma_in ** 2 * 1.0e12
            vmin  = np.min(sigma)
            vmax  = np.max(sigma)
        else:
            sigma = np.ones((ns, nc))
            vmin  = -1
            vmax  =  1

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

                title = (station_labels[ist] if station_labels is not None
                         else f'sta{ist}')
                axes[0, ist].set_title(title, fontsize=9)
                axes[ic, ist].set_xlim([0, n])
                axes[ic, ist].set_ylim([0, n])
                plt.gca().invert_yaxis()

        for ic in range(nc):
            comp = component_labels[ic] if ic < len(component_labels) else str(ic)
            axes[ic, 0].annotate(comp, xy=(0.25, 0.75),
                                 xycoords='axes fraction', ha='right')

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes([0.81, 0.1, 0.02, 0.8])
        cb  = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=norm)
        cb.set_label(
            label='Covariance amplitude ($10^{12}$)' if sigma_in is not None
                  else 'Covariance amplitude',
            fontsize=10)
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_residual_waveform(self, figname='residual_waveform_empiricalCd.png',
                               station_labels=None, delta=1,
                               component_labels=('Vertical', 'Radial', 'Tangential')):
        """
        Plot obs (black), pred (red), and residual (blue) for each seismogram
        in the style of plot_waveform_fit.

        Each station row shows obs overlaid with pred, and the residual
        (obs - pred) below as a separate trace scaled to the same amplitude,
        so the fractional misfit is immediately visible.

        Parameters
        ----------
        figname : str
        station_labels : sequence of str, optional
        delta : float
            Sampling interval in seconds (default 1 s).
        component_labels : sequence of str
        """
        ns  = self.ns
        nc  = self.nc
        nt  = self._obs.shape[2]
        time = np.arange(nt) * delta

        fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc / 3, 1.0 + 0.7 * ns),
                                 sharey=True)
        if nc == 1:
            axes = [axes]

        for ic in range(nc):
            ax = axes[ic]
            ax.set_frame_on(False)

            for is_ in range(ns):
                obs_row  = self._obs[is_, ic]
                pred_row = self._pred[is_, ic]
                res_row  = self._residuals[is_, ic]

                peak = np.amax(np.abs(self._obs[is_]))  # scale by max across all comps
                if peak == 0:
                    continue
                scale = 0.38 / peak

                # obs (black) and pred (red) overlaid at row baseline
                ax.plot(time, obs_row  * scale + is_, c='k',          lw=1.0)
                ax.plot(time, pred_row * scale + is_, c='red',        lw=1.0)
                # residual (blue) shifted down with a thin baseline
                ax.axhline(is_ - 0.42, color='steelblue', lw=0.4, ls='--', alpha=0.5)
                ax.plot(time, res_row  * scale + is_ - 0.42, c='steelblue', lw=0.8)

                obs_pow = np.sum(obs_row ** 2)
                vr = (1 - np.sum(res_row ** 2) / obs_pow) * 100 if obs_pow > 0 else 0.
                if ic == 0:
                    ax.text(0, is_ - 0.75, f'VR={vr:.0f}%',
                            fontsize=7, color='dimgray')

            ax.set_yticks(range(ns))
            ax.tick_params(axis='y', which='both', length=0)
            if ic == 0 and station_labels is not None:
                ax.set_yticklabels(station_labels, fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.set_xlim(0, time[-1])
            ax.set_xticks([0, time[-1]])
            ax.set_xticklabels([0, int(time[-1])])
            ax.set_ylim(-1.0, ns - 0.4)
            ax.plot([0, time[-1]], [-1.0, -1.0], lw=2, c='k')
            ax.set_xlabel('Time (s)')
            comp = component_labels[ic] if ic < len(component_labels) else str(ic)
            ax.set_title(comp)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='k',        lw=1,   label='Obs'),
            Line2D([0], [0], color='red',       lw=1,   label='Pred'),
            Line2D([0], [0], color='steelblue', lw=0.8, label='Residual'),
        ]
        axes[0].legend(handles=legend_elements, loc='upper right',
                       fontsize=7, frameon=False)

        plt.tight_layout()
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
