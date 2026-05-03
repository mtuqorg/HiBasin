#!/usr/bin/env python
"""
Two-phase Bayesian full moment-tensor inversion for the AK 2009-04-07 event.

Phase 1: MCMC with uncorrelated noise (as in EMCEE.FullMomentTensor.AK20090407.py)
         — recovers per-station time shifts.
Phase 2: MCMC with shared correlated noise (Mustać & Tkalčić, 2017, BSSA)
         — fixes time shifts from Phase 1 and samples moment-tensor + noise amplitude.

Two normalised covariance matrices are estimated from the pre-event noise:
  C_dz — shared across all vertical (Z) seismograms
  C_dh — shared across all horizontal (R/T) seismograms

Noise model: two_attenuated_cosine
"""

import os
import numpy as np

from mtuq import read, download_greens
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_beachball
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util.cap import parse_station_codes, Trapezoid

import multiprocessing as mp
from hibasin.misfit.likelihood import MCMC_FullMij
from hibasin.util.covariance_matrix_shared_Cd import covariance_matrix_shared_Cd
from hibasin.misfit.misfit_preparation import shift_greens
from hibasin.visualization.plot_waveform_fit import plot_waveform_fit
from hibasin.visualization.plot_posterior import (
    posterior_distribution_mij,
    posterior_distribution_noise,
    posterior_distribution_timeshift,
)

os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method("fork", force=True)


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Paths & event parameters
    # ------------------------------------------------------------------
    path_data    = '../data/20090407201255351/*.BH[ZRT].sac'
    path_weights = '../data/20090407201255351/weights_surf.dat'
    event_id     = '20090407201255351'
    model        = 'ak135'

    tag = 'sharedCd_2att'

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150,
        capuaf_file=path_weights,
        apply_scaling=False,
    )

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR', 'T'],
    )

    station_id_list = parse_station_codes(path_weights)

    origin = Origin({
        'time':       '2009-04-07T20:12:55.000000Z',
        'latitude':   61.4542,
        'longitude':  -149.7427,
        'depth_in_m': 50000.,
    })
    evdp_in_km = origin.depth_in_m / 1000

    # ------------------------------------------------------------------
    # MPI setup
    # ------------------------------------------------------------------
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # ------------------------------------------------------------------
    # I/O  (rank 0 only)
    # ------------------------------------------------------------------
    if comm.rank == 0:
        print('Reading data...\n')
        data = read(path_data, format='sac',
                    event_id=event_id,
                    station_id_list=station_id_list,
                    tags=['units:m', 'type:displacement'])
        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_sw = data.map(process_sw)

        print('Reading Greens functions...\n')
        greens = download_greens(stations, origin, model)

        print('Processing Greens functions...\n')
        greens_sw = greens.map(process_sw)

        for s in range(len(stations)):
            data_sw[s].resample(1.0)
            greens_sw[s].resample(1.0)
            data_sw[s].station._refresh('delta', 1)
            greens_sw[s].station._refresh('delta', 1)

        stations  = comm.bcast(stations,  root=0)
        data_sw   = comm.bcast(data_sw,   root=0)
        greens_sw = comm.bcast(greens_sw, root=0)

        # --------------------------------------------------------------
        # Shared-covariance noise estimation
        # --------------------------------------------------------------
        print('Estimating shared noise covariance matrices (C_dz, C_dh)...\n')

        data_noise = read(path_data, format='sac',
                          event_id=event_id,
                          station_id_list=station_id_list,
                          tags=['units:m', 'type:displacement'])
        data_noise.sort_by_distance()
        for traces in data_noise:
            traces.resample(1.0)
            traces.station._refresh('delta', 1)

        npts_acf_lag = data_sw[0][0].stats.npts

        noise_estimator = covariance_matrix_shared_Cd(
            origin, data_noise, npts_acf_lag, process_sw,
            noise_length=2000,
            noise_model='two_attenuated_cosine',
        )

        noise_std_sw = noise_estimator.get_noise_std()

        # cov_inv shape: (2, nt, nt)  — compact shared form
        # log_cov_det shape: (ns, nc) — tiled log-det per seismogram
        cov_inv, log_cov_det = noise_estimator.calc_InversionDeterminant_cd()

        print(f'  noise_std shape  : {noise_std_sw.shape}')
        print(f'  cov_inv shape    : {cov_inv.shape}   (2 shared matrices)')
        print(f'  log_cov_det shape: {log_cov_det.shape}\n')

        noise_estimator.plot_noise_series(figname=f'noise_series_{tag}.png')
        noise_estimator.plot_auto_corr_func(figname=f'acf_{tag}.png')
        noise_estimator.plot_data_covariance_matrix(f'covariance_matrix_{tag}.png')

    else:
        stations  = None
        data_sw   = None
        greens_sw = None

    # ------------------------------------------------------------------
    # Phase 1: MCMC with uncorrelated noise — recover time shifts
    # ------------------------------------------------------------------
    if comm.rank == 0:
        print('Phase 1: MCMC sampling with uncorrelated noise (recover time shifts)...\n')
        np.random.seed(1000)

        nwalker  = 512
        nsteps_1 = 10000

        solver1 = MCMC_FullMij(
            misfit_sw, data_sw, greens_sw, noise_std_sw,
            max_noise_parameter=40, M00=1.e13,
            noise_type='uncorrelated',
        )

        print('Phase 1 — ne-%d, ns-%d, nc-%d, ndim-%d'
              % (solver1.ne, solver1.ns, solver1.nc, solver1.ndim))

        init1 = np.random.uniform(-solver1.MAXVAL, solver1.MAXVAL,
                                  (nwalker, solver1.ndim))

        sampler1, pool1 = solver1.get_sampler('emcee', nchains=nwalker)
        sampler1.run_mcmc(init1, nsteps_1, progress=True)
        solver1.cleanup(pool1)

        print(f'Phase 1 acceptance rate: {100 * np.mean(sampler1.acceptance_fraction):.1f}%')
        solver1.save_chains(sampler1, file_path='./', thin=2)

        source_sol_1, noise_sol_1, tau_sol = solver1.get_solution(
            sampler1, warm_up_steps=int(0.5 * nsteps_1), thin=100)

        print(f'Recovered time shifts: {tau_sol}')
        print('Phase 1 best-fit moment tensor:')
        print(source_sol_1.get_dict(0))

        best_mt_1   = source_sol_1.get(0)
        lune_dict_1 = source_sol_1.get_dict(0)

        plot_beachball(
            f'{event_id}_beachball_{tag}_d{evdp_in_km}km_phase1.png',
            best_mt_1, stations, origin)

        plot_waveform_fit(
            best_mt_1.as_vector(), solver1.obs, solver1.greens,
            stations, noise_sol_1, tau_sol,
            f'{event_id}_waveformfit_{tag}_phase1.jpg',
            delta=1, evdp_in_km=evdp_in_km)

        posterior_distribution_mij(
            flat_samples_fname=solver1.chain_fname,
            log_prob_fname=solver1.logprob_fname,
            mt_degree=solver1.ne, thin=2, ratio=0.5,
            figure_fname=f'{event_id}_posterior_source_{tag}_phase1.jpg')

        posterior_distribution_noise(
            flat_samples_fname=solver1.chain_fname,
            mt_degree=solver1.ne, thin=10, ratio=0.5,
            stations=stations,
            figure_fname=f'{event_id}_posterior_noise_{tag}_phase1.jpg')

        posterior_distribution_timeshift(
            solver1, thin=10, ratio=0.5, stations=stations,
            figure_fname=f'{event_id}_posterior_timeshift_{tag}_phase1.jpg')

        # Pre-shift Green's functions with the recovered time shifts before Phase 2
        greens_sw = shift_greens(greens_sw, tau_sol)

    # ------------------------------------------------------------------
    # Phase 2: MCMC with correlated noise — no time shift parameters
    # ------------------------------------------------------------------
    if comm.rank == 0:
        print('Phase 2: MCMC sampling with correlated noise (fixed time shifts)...\n')

        nwalker  = 256
        nsteps_2 = 5000

        solver = MCMC_FullMij(
            misfit_sw, data_sw, greens_sw, noise_std_sw,
            cov_inv=cov_inv, log_cov_det=log_cov_det,
            max_noise_parameter=40, M00=1.e13,
            noise_type='correlated', no_time_shift=True,
        )

        print('Phase 2 — ne-%d, ns-%d, nc-%d, ndim-%d'
              % (solver.ne, solver.ns, solver.nc, solver.ndim))

        init2 = np.random.uniform(-solver.MAXVAL, solver.MAXVAL,
                                  (nwalker, solver.ndim))

        sampler, pool = solver.get_sampler('emcee', nchains=nwalker)
        sampler.run_mcmc(init2, nsteps_2, progress=True)
        solver.cleanup(pool)

        print(f'Phase 2 acceptance rate: {100 * np.mean(sampler.acceptance_fraction):.1f}%')
        solver.save_chains(sampler, file_path='./', thin=2)

        source_sol, noise_sol, _ = solver.get_solution(
            sampler, warm_up_steps=int(0.5 * nsteps_2), thin=100)

        print('Phase 2 best-fit moment tensor:')
        print(source_sol.get_dict(0))

        print('Generating figures...\n')
        best_mt   = source_sol.get(0)
        lune_dict = source_sol.get_dict(0)

        plot_data_greens1(
            f'{event_id}_waveforms_{tag}_d{evdp_in_km}km_phase2.png',
            data_sw, greens_sw, process_sw,
            misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(
            f'{event_id}_beachball_{tag}_d{evdp_in_km}km_phase2.png',
            best_mt, stations, origin)

        # solver.greens is already pre-shifted; pass zeros to avoid double shift
        plot_waveform_fit(
            best_mt.as_vector(), solver.obs, solver.greens,
            stations, noise_sol, np.zeros(2 * solver.ns),
            f'{event_id}_waveformfit_{tag}_phase2.jpg',
            delta=1, evdp_in_km=evdp_in_km)

        posterior_distribution_mij(
            flat_samples_fname=solver.chain_fname,
            log_prob_fname=solver.logprob_fname,
            mt_degree=solver.ne, thin=2, ratio=0.5,
            figure_fname=f'{event_id}_posterior_source_{tag}_phase2.jpg')

        posterior_distribution_noise(
            flat_samples_fname=solver.chain_fname,
            mt_degree=solver.ne, thin=10, ratio=0.5,
            stations=stations,
            figure_fname=f'{event_id}_posterior_noise_{tag}_phase2.jpg')

        print(f'noise_sol : {noise_sol}')
        print(f'tau_sol   : {tau_sol}')
        print('\nFinished\n')
