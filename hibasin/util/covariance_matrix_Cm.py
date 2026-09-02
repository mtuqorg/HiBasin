"""
Earth model uncertainty covariance matrix C_m.

Reference: Phạm and Tkalčić (2021), JGR Solid Earth, doi:10.1029/2021JB022477

Theory error covariance accounts for uncertainty in predicted waveforms caused
by imperfect knowledge of the Earth's velocity structure.  The total data
covariance used in the likelihood is:

    C_total[s,c] = sigma_sc^2 * C_d[s,c]  +  C_m[s,c]

where C_d is the data correlation matrix (covariance_matrix.py), sigma is the
pre-event noise standard deviation, and C_m is constructed here from an ensemble
of Green's functions computed for perturbed 1D Earth models.

C_m depends on the current moment tensor estimate via:

    C_m[s,c,t,t'] = (1/(M-1)) * sum_m  w[m,s,c,t] * w[m,s,c,t']
    w[m,s,c,t]    = sum_e  mij[e] * dev[m,s,c,e,t]
    dev            = GF_ensemble - mean(GF_ensemble, axis=0)  (centered)

where M is the number of perturbed models and ne is the number of elementary
seismograms (6 for full moment tensor).

--------------------------------------------------------------------
Typical workflow
--------------------------------------------------------------------
# Step 1 — compute GF ensemble OUTSIDE this class (expensive, run once):
#
#   For each of M perturbed Earth models, compute MTUQ Green's tensors with
#   the same ProcessData settings as the real data, resample to the same rate,
#   and collect into a numpy array:
#
#       greens_nominal   : (ns, nc, ne, nt)   reference-model GFs
#       greens_ensemble  : (M,  ns, nc, ne, nt) perturbed-model GFs
#
#   Save to disk (np.save / netCDF) for reuse.

# Step 2 — build the model-error covariance object:

#   Same Earth model for every station:
    model_cov = ModelErrorCovariance.from_same_model(greens_ensemble, stations)

#   Different model groups per station (e.g. north / south sub-regions):
    model_cov = ModelErrorCovariance.from_station_groups(
        greens_ensemble_by_group,   # {group_id: (M_g, ns_g, nc, ne, nt)}
        group_station_list,         # {group_id: ['NET.STA1', 'NET.STA2', ...]}
        stations                    # global station list (same order as data)
    )

# Step 3 — compute C_m for a preliminary MT estimate:
#   (from a grid search or a first MCMC run without model error)

    mij_abs = best_mt.as_vector()          # absolute moment tensor (Nm), shape (ne,)
    # or: mij_abs = mij_sampled * solver.M00

    C_m = model_cov.calc_Cm(mij_abs)      # (ns, nc, nt, nt)

# Step 4 — combine with noise covariance and invert once before MCMC:

    cov_d    = noise_estimator.get_covariance_matrix()   # (ns, nc, nt, nt)
    noise_std = noise_estimator.get_noise_std()          # (ns, nc)
    cov_inv, log_cov_det = model_cov.calc_InversionDeterminant(cov_d, noise_std, mij_abs)

# Step 5 — pass to MCMC solver:

    solver = MCMC_FullMij(...,
                          cov_inv=cov_inv, log_cov_det=log_cov_det,
                          noise_type='correlated')
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular


# ---------------------------------------------------------------------------
# Standalone utility — call this outside the class to prepare input arrays
# ---------------------------------------------------------------------------

def compute_greens_deviations(greens_ensemble):
    """
    Mean-center a Green's function ensemble to produce deviations for C_m.

    Centering by the sample mean (rather than by the nominal model) gives the
    unbiased sample covariance with denominator (M-1).

    Parameters
    ----------
    greens_ensemble : ndarray (..., nt)
        Ensemble of Green's functions for M perturbed models.
        Leading dimensions can be (M, ns, nc, ne) or (M, nc, ne), etc.

    Returns
    -------
    dev : ndarray, same shape as greens_ensemble
    """
    return greens_ensemble - greens_ensemble.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class covariance_matrix_Cm:
    """
    Earth model uncertainty covariance matrix C_m.

    Internally stores mean-centered GF deviations per station as a list of
    (nmodels_s, nc, ne, nt) arrays.  C_m is computed on demand for any moment
    tensor via calc_Cm().

    Memory: ~nmodels * ns * nc * ne * nt * 8 bytes
            (e.g. 300 * 12 * 3 * 6 * 350 * 8 ≈ 180 MB
            This is for one 1D model ensemble shared by all stations.  
            Multiple station groups with different ensembles will linearly increase memory usage.
    """

    def __init__(self, dev_by_station, stations, nc, ne, nt):
        """
        Parameters
        ----------
        dev_by_station : list of ndarray, length ns
            dev_by_station[s] has shape (nmodels_s, nc, ne, nt).
            These are mean-centered GF deviations (use compute_greens_deviations).
        stations : list
            Station objects or 'NET.STA' strings, same order as the data.
        nc, ne, nt : int
        """
        self._dev = dev_by_station
        self.stations = stations
        self.ns = len(stations)
        self.nc = nc
        self.ne = ne
        self.nt = nt

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_same_model(cls, greens_ensemble, stations):
        """
        All stations share the same set of perturbed Earth models.

        Parameters
        ----------
        greens_ensemble : ndarray (nmodels, ns, nc, ne, nt)
            MTUQ-processed GFs for M perturbed models.  Apply the same
            ProcessData, resampling, and window as the observed data.
            Compute this outside the class and save to disk for reuse.
        stations : list
            Station list in the same order as data (len = ns).

        Example
        -------
        greens_ensemble = np.load('greens_ensemble.npy')   # (M, ns, nc, ne, nt)
        model_cov = ModelErrorCovariance.from_same_model(greens_ensemble, stations)
        """
        nmodels, ns, nc, ne, nt = greens_ensemble.shape
        assert len(stations) == ns, \
            f"stations length {len(stations)} != greens_ensemble ns={ns}"

        dev_full = compute_greens_deviations(greens_ensemble)   # (M, ns, nc, ne, nt)
        # Each station gets a view into the full deviation array — no copy
        dev_by_station = [dev_full[:, s, :, :, :] for s in range(ns)]
        return cls(dev_by_station, stations, nc, ne, nt)

    @classmethod
    def from_station_groups(cls, greens_ensemble_by_group, group_station_list, stations):
        """
        Different station groups use different perturbed Earth model ensembles.
        Useful when the study area is complicaated enough that a composite of 1D models 
        (i.e., path-specific) is appropriate.

        Parameters
        ----------
        greens_ensemble_by_group : dict {group_id: ndarray (nmodels_g, ns_g, nc, ne, nt)}
            Perturbed GF ensemble for each group.  The second axis (ns_g) must
            match the station order given in group_station_list[group_id].
        group_station_list : dict {group_id: list of str}
            Ordered 'NET.STA' codes for each group, matching the ns_g axis of
            the corresponding array in greens_ensemble_by_group.
        stations : list
            Global station list in the same order as the data (len = ns_total).

        Example
        -------
        model_cov = ModelErrorCovariance.from_station_groups(
            greens_ensemble_by_group = {
                'model1': np.load('greens_model1.npy'),  # (M, 4, nc, ne, nt)
                'model2': np.load('greens_model2.npy'),  # (M, 3, nc, ne, nt)
            },
            group_station_list = {
                'model1': ['IU.BJT', 'IC.MDJ', 'IU.ULN', 'IC.SSE'],
                'model2': ['IU.CHTO', 'IU.TATO', 'IU.MAJO'],
            },
            stations = data.get_stations()
        )
        """
        # Infer nc, ne, nt from first group
        first_ens = next(iter(greens_ensemble_by_group.values()))
        _, _, nc, ne, nt = first_ens.shape

        # Build station_code → (group_id, local_index_within_group)
        station_lookup = {}
        for gid, stat_list in group_station_list.items():
            for local_idx, code in enumerate(stat_list):
                station_lookup[code] = (gid, local_idx)

        # Precompute centered deviations per group (once per group, not per station)
        dev_by_group = {
            gid: compute_greens_deviations(ens)   # (M_g, ns_g, nc, ne, nt)
            for gid, ens in greens_ensemble_by_group.items()
        }

        # Assign a per-station view from its group's deviation array
        dev_by_station = []
        for stat in stations:
            code = _station_code(stat)
            if code not in station_lookup:
                raise KeyError(
                    f"Station '{code}' not found in group_station_list. "
                    f"Available codes: {sorted(station_lookup)}"
                )
            gid, local_idx = station_lookup[code]
            # View into group array: (nmodels_g, nc, ne, nt)
            dev_by_station.append(dev_by_group[gid][:, local_idx, :, :, :])

        return cls(dev_by_station, stations, nc, ne, nt)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def calc_Cm(self, mij):
        """
        Compute theory error covariance C_m for a given moment tensor.

        Parameters
        ----------
        mij : ndarray (ne,)
            Moment tensor in absolute physical units (Nm).
            For MCMC_FullMij use  mij_abs = mij_sampled * solver.M00.

        Returns
        -------
        C_m : ndarray (ns, nc, nt, nt)
        """
        mij = np.asarray(mij, dtype=float)
        C_m = np.zeros((self.ns, self.nc, self.nt, self.nt))

        for s, dev_s in enumerate(self._dev):
            # dev_s : (nmodels_s, nc, ne, nt)
            nmodels = dev_s.shape[0]

            # Step 1: project EGFs onto mij  →  (nmodels, nc, nt)
            # w[m, c, t] = sum_e  mij[e] * dev_s[m, c, e, t]
            w = np.einsum('e, mcet -> mct', mij, dev_s)

            # Step 2: sample covariance outer product  →  (nc, nt, nt)
            # C_m[c, t, t'] = sum_m  w[m,c,t] * w[m,c,t'] / (M-1)
            C_m[s] = np.einsum('mct, mcp -> ctp', w, w) / (nmodels - 1)

        return C_m

    def calc_effective_cov(self, cov_d, noise_std, mij):
        """
        Compute the noise-normalised effective correlation matrix:

            C_eff[s,c] = C_d[s,c]  +  C_m[s,c] / noise_std[s,c]^2

        This is the matrix to invert and pass to the MCMC solver when using
        noise_type='correlated' in likelihood.py WITHOUT any change to that file.
        The existing likelihood then computes:

            log L = -½ [ r^T (noise_amp^2 · C_eff)^{-1} r
                        + log|noise_amp^2 · C_eff| ]

        which equals the exact formula at noise_amp == noise_std (where the
        posterior is constrained by the data).  The noise_amp parameter remains
        free and retains its usual diagnostic meaning.

        Parameters
        ----------
        cov_d : ndarray (ns, nc, nt, nt)
            Normalised noise correlation matrix (diagonal = 1) from
            covariance_matrix.get_covariance_matrix().
        noise_std : ndarray (ns, nc)
            Pre-event noise standard deviation from get_noise_std().
        mij : ndarray (ne,)
            Moment tensor in absolute units (Nm).

        Returns
        -------
        C_eff : ndarray (ns, nc, nt, nt)
        """
        C_m = self.calc_Cm(mij)
        return cov_d + C_m / noise_std[:, :, None, None] ** 2

    def calc_InversionDeterminant(self, cov_d, noise_std, mij):
        """
        Invert C_eff = C_d + C_m / noise_std^2 via Cholesky decomposition.

        The result is directly compatible with likelihood.py (noise_type='correlated')
        without any modification to that file.  noise_amp remains a free MCMC
        parameter; the approximation is exact when noise_amp == noise_std.

        Parameters
        ----------
        cov_d : ndarray (ns, nc, nt, nt)
            Normalised noise correlation matrix (diagonal = 1).
        noise_std : ndarray (ns, nc)
            Pre-event noise standard deviation.
        mij : ndarray (ne,)
            Moment tensor in absolute units (Nm).

        Returns
        -------
        cov_inv : ndarray (ns, nc, nt, nt)
            C_eff^{-1} per station/component.
        log_cov_det : ndarray (ns, nc)
            sum(log(diag(L))) where L L^T = C_eff.
            Equals 0.5 * log|C_eff|.  Consistent with
            covariance_matrix.calc_InversionDeterminant_cd convention.
        """
        C_eff = self.calc_effective_cov(cov_d, noise_std, mij)
        nt = self.nt
        cov_inv = np.zeros_like(C_eff)
        log_cov_det = np.zeros((self.ns, self.nc))

        for s in range(self.ns):
            for c in range(self.nc):
                cov = C_eff[s, c]
                covL = cholesky(cov, lower=True)
                log_cov_det[s, c] = np.sum(np.log(np.diag(covL)))
                covL_inv = solve_triangular(covL, np.eye(nt), lower=True)
                cov_inv[s, c] = covL_inv.T @ covL_inv

        return cov_inv, log_cov_det

    def calc_Cm_eigen(self, mij):
        """
        Eigendecomposition of C_m for use with likelihood_Cm.py.

        Returns the eigenvalues and eigenvectors of C_m[s,c] = V Λ Vᵀ so that
        the exact likelihood for C_total = k_s²σ²I + C_m can be evaluated as:

            rᵀ C_total⁻¹ r  =  Σᵢ (Vᵀr)ᵢ² / (k_s²σ² + λᵢ)
            log|C_total|    =  Σᵢ log(k_s²σ² + λᵢ)

        k_s² appears only as an additive offset to λᵢ, not as a multiplicative
        scale on C_m.  This avoids the bias present in calc_InversionDeterminant.

        Parameters
        ----------
        mij : ndarray (ne,)
            Moment tensor in absolute physical units (Nm).
            For MCMC_FullMij use  mij_abs = mij_sampled * solver.M00.

        Returns
        -------
        eigvals : ndarray (ns, nc, nt)
            Eigenvalues λᵢ of C_m per station and component.
        eigvecs : ndarray (ns, nc, nt, nt)
            Eigenvector matrix V (columns are eigenvectors) per station
            and component.  np.linalg.eigh guarantees V is orthonormal.
        """
        C_m = self.calc_Cm(mij)    # (ns, nc, nt, nt)
        eigvals = np.zeros((self.ns, self.nc, self.nt))
        eigvecs = np.zeros((self.ns, self.nc, self.nt, self.nt))
        for s in range(self.ns):
            for c in range(self.nc):
                lam, V = np.linalg.eigh(C_m[s, c])
                eigvals[s, c] = lam
                eigvecs[s, c] = V
        return eigvals, eigvecs

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self):
        """Print a brief summary of the stored ensemble sizes."""
        print(f"ModelErrorCovariance: ns={self.ns}, nc={self.nc}, "
              f"ne={self.ne}, nt={self.nt}")
        for s, (stat, dev_s) in enumerate(zip(self.stations, self._dev)):
            code = _station_code(stat)
            print(f"  [{s:2d}] {code:12s}  nmodels={dev_s.shape[0]}")

    def plot_model_covariance_matrix(self, mij, figname, noise_std=None):
        """
        Plot C_m for each station and component.

        Parameters
        ----------
        mij : ndarray (ne,)
            Moment tensor in absolute physical units (Nm).
        figname : str
            Output file path (e.g. 'Cm_matrix.png').
        noise_std : ndarray (ns, nc) or None
            If given, C_m is divided by noise_std^2 (dimensionless ratio).
        """
        cov_m = self.calc_Cm(mij)          # (ns, nc, nt, nt)
        ns, nc, nt, _ = cov_m.shape
        comps = ['Z', 'R', 'T'][:nc]

        if noise_std is not None:
            plot_data = cov_m / noise_std[:, :, None, None] ** 2
        else:
            plot_data = cov_m

        fig, axes = plt.subplots(
            nc, ns,
            sharex=True, sharey=True,
            figsize=(min(9, ns), 2.5),
        )
        if ns == 1:
            axes = axes[:, np.newaxis]

        cm = plt.get_cmap('jet')

        for ist in range(ns):
            for ic in range(nc):
                panel = plot_data[ist, ic]
                pmax = np.max(np.abs(panel))
                axes[ic, ist].imshow(panel, vmin=-pmax, vmax=pmax,
                                     cmap=cm, aspect='auto')

                # Max in top-left, min in bottom-right
                axes[ic, ist].text(
                    0.02, 0.97, f'{panel.max():.1e}',
                    transform=axes[ic, ist].transAxes,
                    ha='left', va='top', fontsize=5, color='white',
                )
                axes[ic, ist].text(
                    0.98, 0.03, f'{panel.min():.1e}',
                    transform=axes[ic, ist].transAxes,
                    ha='right', va='bottom', fontsize=5, color='white',
                )

                axes[0, ist].set_title(
                    _station_code(self.stations[ist]).split('.')[-1], fontsize=9,
                )
                if ist == 0:
                    axes[ic, ist].set_ylabel(comps[ic], fontsize=9)
                if ic == nc - 1:
                    axes[ic, ist].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved → {figname}")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _station_code(station):
    """Return 'NET.STA' string from a station object or bare string."""
    if isinstance(station, str):
        return station
    try:
        return f"{station.network}.{station.station}"
    except AttributeError:
        return str(station)
