#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_beachball
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util.cap import parse_station_codes, Trapezoid

import multiprocessing as mp
import sys
from hibasin.misfit.likelihood import MCMC_FullMij
from hibasin.util.covariance_matrix_Cd import covariance_matrix_Cd
from hibasin.misfit.misfit_preparation import shift_greens
from hibasin.util.math import cc_optimal_shifts
from hibasin.visualization.plot_waveform_fit import plot_waveform_fit
from hibasin.visualization.plot_posterior import posterior_distribution_mij, posterior_distribution_noise, posterior_distribution_timeshift

os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method("fork", force=True)

if __name__=='__main__':
    # 
    path_data=    '../data/20170903033001000/*.BH[ZRT].sac'
    path_weights= '../data/20170903033001000/weights_surf.dat'
    CPS_database= '../data/grn_2017_2d/mdj3/'
    event_id=     '20170903033001000'
    model=        'mdj3'

    #
    # Surface wave measurements is used here.
    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.02,
        freq_max=0.05,
        pick_type='CPS_metadata',
        CPS_database=CPS_database,
        window_type='surface_wave',
        window_length=350,
        capuaf_file=path_weights,
        apply_scaling = False
        )
    
    #
    # For our objective function, we will use surface wave
    # contribution only
    #
    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-11,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T']
        )

    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #
    station_id_list = parse_station_codes(path_weights)

    #
    # Next, we specify the source-time function
    #
    wavelet = Trapezoid(
        magnitude=5.1)

    #
    # Origin time and location will be fixed. 
    #
    origin = Origin({
        'time': '2017-09-03T03:30:01.760000Z',
        'latitude': 41.3,
        'longitude': 129.078,
        'depth_in_m': 500.,
        })
    evdp_in_km = origin.depth_in_m/1000


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
    db = open_db(CPS_database,  format='CPS', model=model)
    greens = db.get_greens_tensors(stations, origin)

    print('Processing Greens functions...\n')
    greens.convolve(wavelet)
    greens_sw = greens.map(process_sw)

    ##resample the data and greens
    for s in range(len(stations)):
        data_sw[s].resample(1)
        greens_sw[s].resample(1)
        #update the delta in dataset.station
        data_sw[s].station._refresh('delta',1)
        greens_sw[s].station._refresh('delta',1)

    ##estimate the noise strength and covariance matrix
    data_noise = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:displacement'])
    data_noise.sort_by_distance()
    for traces in data_noise:
        traces.resample(1)
    npts_acf_lag = data_sw[0][0].stats.npts
    noise_estimator = covariance_matrix_Cd(origin, data_noise, npts_acf_lag, process_sw, 
                                            noise_length=3000, noise_model='uncorrelated')
    noise_std_sw = noise_estimator.get_noise_std()

    #    
    # The main computational work starts now
    #
    print('Starting MCMC sampling...\n')
    np.random.seed(1000)

    nwalker = 512
    nsteps = 10000
    n_burnin = 500    # short burn-in for CC warm start

    # Create the MCMC solver — ndim and all dimensions are derived internally
    solver = MCMC_FullMij(misfit_sw, data_sw, greens_sw,
                        noise_std_sw, max_noise_parameter=400,
                        M00=1.0e15, noise_type='uncorrelated')

    print('Important parameters: ne-%d, ns-%d, nc-%d, ndim-%d'
            % (solver.ne, solver.ns, solver.nc, solver.ndim))

    sampler, pool = solver.get_sampler('emcee', nchains=nwalker)

    # ── Phase 1: short burn-in to get reference MT ───────────────────────
    print('Phase 1: burn-in (%d steps) ...' % n_burnin)
    init_uniform = np.random.uniform(-solver.MAXVAL, solver.MAXVAL,
                                     (nwalker, solver.ndim))
    sampler.run_mcmc(init_uniform, n_burnin, progress=False)

    m_best  = sampler.flatchain[np.argmax(sampler.get_log_prob(flat=True))]
    ref_mij = solver._params_to_mij(m_best)

    # ── CC warm start ─────────────────────────────────────────────────────
    tau_cc = cc_optimal_shifts(solver, ref_mij)
    print('CC-optimal shifts (s): ZR=%s  T=%s'
          % (np.round(tau_cc[0::2], 2), np.round(tau_cc[1::2], 2)))

    m_tau_raw        = (tau_cc - solver.time_shift_scale2) / solver.time_shift_scale1
    m_tau_raw_active = m_tau_raw[solver.timeshift_mask]

    scatter_raw = 0.5 / solver.time_shift_scale1
    init2 = np.random.uniform(-solver.MAXVAL, solver.MAXVAL, (nwalker, solver.ndim))
    init2[:, solver.ne + solver.ns:] = (
        m_tau_raw_active[np.newaxis, :]
        + np.random.normal(0, scatter_raw, (nwalker, m_tau_raw_active.size)))
    init2 = np.clip(init2, -solver.MAXVAL, solver.MAXVAL)

    # ── Phase 2: full MCMC from CC warm start ───────────────────────────────
    print('Phase 2: full MCMC (%d steps) ...' % nsteps)
    sampler.reset()
    state = sampler.run_mcmc(init2, nsteps, progress=True)
    solver.cleanup(pool)

    ## Print acceptance fraction for diagnosis
    acceptance_rate = 100 * np.mean(sampler.acceptance_fraction)
    print ('Average acceptance rate: %d' % acceptance_rate + '%')
        
    ##write the samples into files
    solver.save_chains(sampler, file_path='./', thin=2)
        
    #
    # Generate figures and save results
    #
    
    ## Extract the chain for inspection
    source_sol, noise_sol, tau_sol = solver.get_solution(sampler, warm_up_steps=int(0.5*nsteps), thin=100)
    # dictionary of Mij parameters
    print("The best mt:")
    print(source_sol.get_dict(0))

    print('Generating figures...\n')
    best_mt = source_sol.get(0)
    lune_dict = source_sol.get_dict(0)
    greens_sw = shift_greens(greens_sw, tau_sol)
    plot_data_greens1(event_id+'_Mij_waveforms_sw_d%skm_noise_cd.png' % evdp_in_km,
        data_sw, greens_sw, process_sw, 
        misfit_sw, stations, origin, best_mt, lune_dict)

    plot_beachball(event_id+'_Mij_beachball_sw_d%skm_noise_cd.png' % evdp_in_km,
        best_mt, stations, origin)
    
    plot_waveform_fit(best_mt.as_vector(), solver.obs, solver.greens, stations, noise_sol, tau_sol, \
                        event_id+'_Waveformfit_mean.jpg', delta=1, evdp_in_km=evdp_in_km)

    #
    # Plot the posterior distribution
    posterior_distribution_mij(flat_samples_fname=solver.chain_fname,log_prob_fname=solver.logprob_fname, 
                                mt_degree=solver.ne, thin=2,ratio=0.5, figure_fname=event_id+"_Posterior_source_parameter.jpg")
    posterior_distribution_noise(flat_samples_fname=solver.chain_fname, mt_degree=solver.ne, thin=10, 
                                    ratio=0.5,stations=stations, figure_fname=event_id+'_Posterior_data_noise.jpg')
    posterior_distribution_timeshift(solver, thin=10, ratio=0.5,stations=stations, figure_fname=event_id+'_Posterior_timeshift.jpg')
    print(noise_sol)
    print(tau_sol)
    print('\nFinished\n')

