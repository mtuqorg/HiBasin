"""
MCMC solvers with exact model-error covariance (Phạm & Tkalčić 2021).

The total covariance used in the likelihood is:

    C_total[s,c] = k_s[s]² · σ[s,c]² · I  +  C_m[s,c]

where C_m is the model-error covariance computed from a Green's function
ensemble via covariance_matrix_Cm.calc_Cm_eigen().  Time shifts are NOT
sampled — C_m captures model-timing variability through off-diagonal
structure, so including time-shift parameters would double-count it.

The exact quadratic form and log-determinant are evaluated via the
eigendecomposition C_m = V Λ Vᵀ (precomputed once):

    rᵀ C_total⁻¹ r  =  Σᵢ (Vᵀr)ᵢ² / (k_s²σ² + λᵢ)
    log|C_total|    =  Σᵢ log(k_s²σ² + λᵢ)

k_s scales only the data-noise floor σ²I; C_m eigenvalues λᵢ are fixed
regardless of k_s.  This avoids the bias present in the pre-computed
C_eff = I + C_m/σ² approach (where k_s² would scale C_m as well).

Sampled dimensions: ne (source params) + ns (per-station noise amplitudes).

Usage
-----
    eigvals, eigvecs = model_cov.calc_Cm_eigen(mij_prior)

    solver = MCMC_FullMij(data_sw, greens_sw, noise_std_sw,
                          Cm_eigvals=eigvals, Cm_eigvecs=eigvecs,
                          max_noise_parameter=40, M00=1e14)
    sampler, pool = solver.get_sampler('emcee', nchains=512)
    sampler.run_mcmc(p0, nsteps, progress=True)
    solver.cleanup(pool)
    source_sol, noise_sol, tau_sol = solver.get_solution(sampler,
                                         warm_up_steps=3000, thin=200)
"""

import os
import emcee
import numpy as np
import multiprocessing as mp
import multiprocessing.shared_memory

from mtuq.grid.moment_tensor import to_mt
from mtuq.grid.force import to_force
from mtuq.grid import UnstructuredGrid
from mtuq.misfit.waveform import level2
from mtuq.util.math import to_mij, to_rho, to_rtp

from hibasin.util.math import to_lune, Tashiro2MT6, ned2rtp
from hibasin.misfit.misfit_preparation import to_numpy_arrays

os.environ["OMP_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Module-level shared data — isolated from likelihood.py's shared_data dict
# ---------------------------------------------------------------------------

shared_data = {}


def pool_initializer(shm_name, shape, dtype_str, eigvals):
    """Worker initializer: map eigvecs shared memory and store eigvals."""
    existing_shm = mp.shared_memory.SharedMemory(name=shm_name)
    shared_data['Cm_eigvecs'] = np.ndarray(shape, dtype=np.dtype(dtype_str),
                                            buffer=existing_shm.buf)
    shared_data['Cm_eigvals'] = eigvals   # small array, safe to fork-copy
    shared_data['shm'] = existing_shm


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _MCMC_Cm_Base:
    """
    Base class for MCMC solvers with exact model-error covariance.

    Subclasses must implement:
        _params_to_mij(m)         — map sampled vector → 6-element mij
        get_solution(sampler, …)  — extract physical source solution
        save_chains(sampler, …)   — save and transform chains to disk
    """

    _apply_m00 = False

    def __init__(self, data_sw, greens_sw, noise_std_sw,
                 Cm_eigvals, Cm_eigvecs,
                 max_noise_parameter=100,
                 M00=None):
        """
        Parameters
        ----------
        data_sw : mtuq Dataset
            Processed observed waveforms.
        greens_sw : mtuq GreensTensorList
            Processed Green's functions (same ProcessData as data_sw).
        noise_std_sw : ndarray (ns, 3)
            Pre-event noise standard deviation from get_noise_std().
            Column order is fixed Z=0, R=1, T=2 regardless of which
            components are active.
        Cm_eigvals : ndarray (ns, 3, nt)
            Eigenvalues of C_m from covariance_matrix_Cm.calc_Cm_eigen().
        Cm_eigvecs : ndarray (ns, 3, nt, nt)
            Eigenvectors of C_m from covariance_matrix_Cm.calc_Cm_eigen().
        max_noise_parameter : float
            Upper bound for the per-station noise multiplier k_s.
        M00 : float or None
            Reference moment scale.  When set and _apply_m00 is True,
            greens are pre-scaled so that sampled MT params are O(1).
        """
        self.MAXVAL = 3600

        self.obs, self.greens, self.weight_mask = to_numpy_arrays(data_sw, greens_sw)

        if M00 is not None and self._apply_m00:
            self.M00 = M00
            self.greens *= self.M00
        else:
            self.M00 = None

        self.ns, self.nc, self.ne, self.nt = self.greens.shape
        _, self.delta = level2._get_time_sampling(data_sw)

        # Map active components to fixed noise_std column order (Z=0, R=1, T=2).
        _comp_col = {'Z': 0, 'R': 1, 'T': 2}
        _active_cols = [_comp_col[c] for c in level2._get_components(data_sw)
                        if c in _comp_col]
        self.noise_std = noise_std_sw[:, _active_cols].astype(np.float32)

        # Noise multiplier encoding: k_s ∈ [0, max_noise_parameter]
        # maps from MCMC parameter m ∈ [-MAXVAL, MAXVAL]
        self.max_noise = max_noise_parameter
        self.MAXVAL2 = self.MAXVAL * 2
        self.noise_scale1 = max_noise_parameter / self.MAXVAL2
        self.noise_scale2 = self.noise_scale1 * self.MAXVAL

        # No time shifts: ndim = ne (source) + ns (noise per station)
        self.no_time_shift = True
        self.ndim = self.ne + self.ns
        self.noise_type = 'correlated_Cm'

        # Slice eigenvectors/values to active components and put eigvecs in
        # shared memory (ns × nc × nt × nt × 8 bytes ≈ 15 MB for typical runs)
        self.Cm_eigvals = Cm_eigvals[:, _active_cols, :].astype(np.float64)
        eigvecs_active = Cm_eigvecs[:, _active_cols, :, :].astype(np.float64)

        try:
            self.shm = mp.shared_memory.SharedMemory(
                create=True, size=eigvecs_active.nbytes)
            buf = np.ndarray(eigvecs_active.shape, dtype=eigvecs_active.dtype,
                             buffer=self.shm.buf)
            buf[:] = eigvecs_active[:]
            self.eigvecs_shape = buf.shape
            self.eigvecs_dtype = buf.dtype
        except Exception:
            if hasattr(self, 'shm'):
                self.shm.close()
                self.shm.unlink()
            raise

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------

    def _log_prior(self, m):
        return 0.0 if np.all((-self.MAXVAL <= m) & (m <= self.MAXVAL)) else -np.inf

    # ------------------------------------------------------------------
    # Parameterization hook
    # ------------------------------------------------------------------

    def _params_to_mij(self, m):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def log_prob(self, m):
        if not np.isfinite(self._log_prior(m)):
            return -np.inf
        mij = self._params_to_mij(m)
        pred = np.einsum('scet,e->sct', self.greens, mij)
        amp = m[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2
        return self._likelihood(pred, amp)

    def _likelihood(self, pred, amp):
        """
        Exact log-likelihood for C_total = k_s²σ²I + C_m.

        Uses the precomputed eigendecomposition C_m = V Λ Vᵀ so that
        k_s only rescales the data-noise floor σ²I, not C_m.
        """
        noise_amp = self.noise_std * amp[:, None]          # (ns, nc) = k_s · σ
        res = self.obs - pred                               # (ns, nc, nt)
        eigvecs = shared_data['Cm_eigvecs']                # (ns, nc, nt, nt)
        eigvals = shared_data['Cm_eigvals']                # (ns, nc, nt)

        lp1 = np.zeros((self.ns, self.nc))
        lp2 = np.zeros((self.ns, self.nc))
        for s in range(self.ns):
            for c in range(self.nc):
                denom = noise_amp[s, c] ** 2 + eigvals[s, c]  # k_s²σ² + λᵢ, shape (nt,)
                Vt_r  = eigvecs[s, c].T @ res[s, c]            # Vᵀr, shape (nt,)
                lp1[s, c] = np.sum(Vt_r ** 2 / denom)         # rᵀ C_total⁻¹ r
                lp2[s, c] = np.sum(np.log(denom))              # log|C_total|

        result = lp1 + lp2
        result[self.weight_mask.astype(bool)] = 0.0
        return -0.5 * np.sum(result)

    # ------------------------------------------------------------------
    # Sampler, cleanup, diagnostics
    # ------------------------------------------------------------------

    def get_sampler(self, method='emcee', nchains=512):
        ctx = mp.get_context("fork")
        pool = ctx.Pool(initializer=pool_initializer,
                        initargs=(self.shm.name, self.eigvecs_shape,
                                  self.eigvecs_dtype.str, self.Cm_eigvals))
        if method == 'emcee':
            sampler = emcee.EnsembleSampler(nchains, self.ndim, self.log_prob, pool=pool)
            print('Creating emcee sampler (Cm exact): chains=%d  ndim=%d'
                  % (nchains, self.ndim))
        else:
            raise ValueError(f"Unknown sampler method: {method}")
        return sampler, pool

    def reset(self):
        if self.M00 is not None:
            self.greens /= self.M00

    def cleanup(self, pool):
        pool.close()
        pool.join()
        self.shm.close()
        self.shm.unlink()
        print("Shared memory cleaned up.")
        self.reset()

    def diagnose(self, sampler):
        tau = sampler.get_autocorr_time(tol=0)
        print('\nAutocorrelation time for each coordinate:\n    ', tau)

    # ------------------------------------------------------------------
    # Shared helpers for subclass get_solution / save_chains
    # ------------------------------------------------------------------

    def _noise_solution(self, m_sol):
        """Decode noise multipliers from posterior mean MCMC vector."""
        return m_sol[self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2

    def _transform_noise_chains(self, samples):
        """Convert noise columns from encoded to physical units in-place copy."""
        samples = samples.copy()
        samples[:, self.ne:self.ne + self.ns] = (
            samples[:, self.ne:self.ne + self.ns] * self.noise_scale1 + self.noise_scale2)
        return samples

    def _save_chains_core(self, sampler, flat_samples, file_path, tag, thin):
        self.chain_fname = (file_path
                            + 'MCMC_sampling_%s_correlated_Cm_model.npy' % tag)
        np.save(self.chain_fname, flat_samples)
        self.logprob_fname = (file_path
                              + 'MCMC_sampling_%s_correlated_Cm_log_prob.npy' % tag)
        log_prob = sampler.get_log_prob(discard=0, thin=thin, flat=True)
        np.save(self.logprob_fname, log_prob)

    def get_map_mij(self, sampler, warm_up_steps=0, thin=1):
        """
        Return MAP moment-tensor coefficients in solver internal units
        (compatible with solver.greens already scaled by M00).
        """
        flat  = sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        m_sol = np.mean(flat, axis=0)
        return self._params_to_mij(m_sol)

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        raise NotImplementedError

    def save_chains(self, sampler, file_path='./', thin=1):
        raise NotImplementedError


# =============================================================================
# Moment tensor parameterizations
# =============================================================================

class MCMC_FullMij(_MCMC_Cm_Base):
    """
    Full moment tensor with exact model-error covariance.

    Sampled parameters: m[0:6] MT components, m[6:6+ns] noise multipliers.
    ndim = 6 + ns.  Green's functions are pre-scaled by M00.
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
        noise_sol = self._noise_solution(m_sol)
        return source_solution, noise_sol, np.zeros(2 * self.ns)

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        flat = self._transform_noise_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'mij', thin)


class MCMC_DeviatoricMij(_MCMC_Cm_Base):
    """
    Deviatoric moment tensor with exact model-error covariance.

    Mrr is constrained as -(Mtt + Mpp).
    Sampled parameters: m[0:5] MT components, m[5:5+ns] noise multipliers.
    ndim = 5 + ns.  Green's functions are pre-scaled by M00.
    """

    _apply_m00 = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override ne/ndim: sample 5 MT params, not 6
        self.ne = 5
        self.ndim = self.ne + self.ns

    def _params_to_mij(self, m):
        mij = np.empty(6, dtype=float)
        mij[0] = -m[0] - m[1]   # Mrr = -(Mtt + Mpp)
        mij[1:] = m[:5]
        return mij

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        mij5 = m_sol[:5] * self.M00
        mij  = np.array([-mij5[0] - mij5[1],
                          mij5[0], mij5[1], mij5[2], mij5[3], mij5[4]])
        rho, v, w, kappa, sigma, h = to_lune(mij)
        source_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)
        noise_sol = self._noise_solution(m_sol)
        return source_solution, noise_sol, np.zeros(2 * self.ns)

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        flat = self._transform_noise_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'mij_deviatoric', thin)


class MCMC_TT2015(_MCMC_Cm_Base):
    """
    Tape & Tape (2015) lune parameterization with exact model-error covariance.

    Sampled dimensions (ne=6): v, w, kappa, sigma, h, Mw (all encoded to
    [-MAXVAL, MAXVAL]).  ndim = 6 + ns.

    Parameters
    ----------
    mw_min, mw_max : float
        Moment-magnitude search range (default 4.0–6.0).

    Citation
    --------
    Tape, W., & Tape, C. (2015). A uniform parametrization of moment tensors.
    Geophysical Journal International, 202(3), 2074–2081.
    """

    def __init__(self, *args, mw_min=4.0, mw_max=6.0, **kwargs):
        super().__init__(*args, **kwargs)
        if mw_min >= mw_max:
            raise ValueError(f'mw_min ({mw_min}) must be less than mw_max ({mw_max})')
        self.mw_min = mw_min
        self.mw_max = mw_max

    def _decode_mw(self, val):
        return (val + self.MAXVAL) / self.MAXVAL2 * (self.mw_max - self.mw_min) + self.mw_min

    def _params_to_mij(self, m):
        v     = m[0] / 10800
        w     = m[1] * np.pi / 9600
        kappa = (m[2] + self.MAXVAL) / 20
        sigma = m[3] / 40
        h     = (m[4] + self.MAXVAL) / 7200
        rho   = to_rho(self._decode_mw(m[5]))
        return to_mij(rho, v, w, kappa, sigma, h)

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        v     = m_sol[0] / 10800
        w     = m_sol[1] * np.pi / 9600
        kappa = (m_sol[2] + self.MAXVAL) / 20
        sigma = m_sol[3] / 40
        h     = (m_sol[4] + self.MAXVAL) / 7200
        rho   = to_rho(self._decode_mw(m_sol[5]))

        source_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)
        noise_sol = self._noise_solution(m_sol)
        return source_solution, noise_sol, np.zeros(2 * self.ns)

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        to_rho_vec = np.vectorize(to_rho)
        flat[:, :6] = np.column_stack((
            flat[:, 0] / 10800,
            flat[:, 1] * np.pi / 9600,
            (flat[:, 2] + self.MAXVAL) / 20,
            flat[:, 3] / 40,
            (flat[:, 4] + self.MAXVAL) / 7200,
            to_rho_vec(self._decode_mw(flat[:, 5])),
        ))
        flat = self._transform_noise_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'tt2015', thin)


class MCMC_Tashiro(_MCMC_Cm_Base):
    """
    Tashiro parameterization with exact model-error covariance.

    Sampled parameters (ne=6): x1–x5 mapped to (0,1) and Mw encoded.
    ndim = 6 + ns.

    Parameters
    ----------
    mw_min, mw_max : float
        Moment-magnitude search range (default 4.0–6.0).

    Citation
    --------
    Stähler, S. C., & Sigloch, K. (2014). Fully probabilistic seismic source
    inversion — Part 1: Efficient parameterisation. Solid Earth, 5(2), 1055–1069.
    """

    def __init__(self, *args, mw_min=4.0, mw_max=6.0, **kwargs):
        super().__init__(*args, **kwargs)
        if mw_min >= mw_max:
            raise ValueError(f'mw_min ({mw_min}) must be less than mw_max ({mw_max})')
        self.mw_min = mw_min
        self.mw_max = mw_max

    def _decode_mw(self, val):
        return (val + self.MAXVAL) / self.MAXVAL2 * (self.mw_max - self.mw_min) + self.mw_min

    def _params_to_mij(self, m):
        mc = m[:6].copy()
        mc[:5] = (mc[:5] + self.MAXVAL) / self.MAXVAL2
        mc[5]  = self._decode_mw(m[5])
        return ned2rtp(Tashiro2MT6(mc))

    def get_solution(self, emcee_sampler, warm_up_steps, thin):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)

        xi  = (self.MAXVAL + m_sol[:5]) / self.MAXVAL2
        mw  = self._decode_mw(m_sol[5])
        mij = ned2rtp(Tashiro2MT6(np.concatenate([xi, [mw]])))
        rho, v, w, kappa, sigma, h = to_lune(mij)

        source_solution = UnstructuredGrid(
            dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v, w, kappa, sigma, h),
            callback=to_mt)
        noise_sol = self._noise_solution(m_sol)
        return source_solution, noise_sol, np.zeros(2 * self.ns)

    def save_chains(self, sampler, file_path='./', thin=1):
        flat = sampler.get_chain(discard=0, thin=thin, flat=True)
        flat[:, :6] = np.column_stack((
            (self.MAXVAL + flat[:, :5]) / self.MAXVAL2,
            self._decode_mw(flat[:, 5]),
        ))
        flat = self._transform_noise_chains(flat)
        self._save_chains_core(sampler, flat, file_path, 'tashiro', thin)
