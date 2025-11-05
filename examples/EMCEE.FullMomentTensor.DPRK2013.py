#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_beachball, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.grid import UnstructuredGrid
from mtuq.grid.moment_tensor import to_mt
from mtuq.util.math import to_mij, to_rho
from mtuq.util.cap import taper

import multiprocessing as mp
import emcee
import sys
from src.misfit.likelihood import MCMC_SOLVER
from src.util.covariance_matrix import covariace_matrix
from src.util.math import exponential_covariance, calc_InversionDeterminant_cd
from src.util.data_selection import data_noise_estimate_uncorrelated
from src.util.misfit_preparation import shift_greens, shift_data,  misfit_preparation
from src.visualization.plot_waveform_fit import plot_waveform_fit
from src.visualization.plot_posterior import posterior_distribution_mij, posterior_distribution_noise, posterior_distribution_timeshift
from obspy.signal.filter import bandpass
from multiprocessing import shared_memory

os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method("fork", force=True)

if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   

    path_data=    fullpath('/Users/u7091895/Documents/Research/BayMTI/HiBaysin/data/20130212025751000/*.BH[ZRT].sac')
    path_weights= fullpath('/Users/u7091895/Documents/Research/BayMTI/HiBaysin/data/20130212025751000/weights_surf.dat')
    event_id=     '20130212025751000'
    model=        'mdj3'

    #
    # Surface wave measurements will be made separately
    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.02,
        freq_max=0.05,
        # pick_type='taup',
        # taup_model=model,
        pick_type='CPS_metadata',
        CPS_database='/Users/u7091895/Documents/Research/BayMTI/HiBaysin/data/grn_2013_2d/',
        CPS_model=model,
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
        time_shift_min=-5.,
        time_shift_max=+5.,
        time_shift_groups=['ZR','T'],
        )

    #
    # User-supplied weights control how much each station contributes to the
    # objective function
    #
    station_id_list = parse_station_codes(path_weights)

    #
    # Next, we specify the moment tensor grid and source-time function
    #
    wavelet = Trapezoid(
        magnitude=4.5)

    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2013-02-12T02:57:51.272500Z',
        'latitude': 41.2921,
        'longitude': 129.0730,
        'depth_in_m': 500.,
        })
    evdp_in_km = origin.depth_in_m/1000

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    #
    # The main I/O work starts now
    #

    if comm.rank==0:
        print('Reading data...\n')
        data = read(path_data, format='sac', 
            event_id=event_id,
            station_id_list=station_id_list,
            tags=['units:m', 'type:displacement']) 

        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_sw = data.map(process_sw)

        #custom allowable time shift
        shift = np.zeros(2*len(stations))
        #Rayleigh
        shift[::2] = 10
        #Love
        shift[1::2] = -2
        shift[-1] = 5
        shift[-3] = 5
        data_sw = shift_data(data_sw, shift)


        print('Reading Greens functions...\n')
        # greens = download_greens_tensors(stations, origin, model)
        db = open_db('/Users/u7091895/Documents/Research/BayMTI/HiBaysin/data/grn_2013_2d/mdj3',  format='CPS', model=model)
        greens = db.get_greens_tensors(stations, origin)

        print('Processing Greens functions...\n')
#         greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)

        ##resample the data and greens
        for s in range(len(stations)):
            data_sw[s].resample(1.0)
            greens_sw[s].resample(1.0)
            #update the delta in dataset.station
            data_sw[s].station._refresh('delta',1)
            greens_sw[s].station._refresh('delta',1)

        stations = comm.bcast(stations, root=0)
        data_sw = comm.bcast(data_sw, root=0)
        greens_sw = comm.bcast(greens_sw, root=0)

        ##estimate the noise strength and covariance matrix
        data_noise = read(path_data, format='sac', 
                event_id=event_id,
                station_id_list=station_id_list,
                tags=['units:m', 'type:displacement'])
        for traces in data_noise:
            traces.resample(1.0)
        npts_acf_lag = data_sw[0][0].stats.npts
        noise_estimator = covariace_matrix(origin, data_noise, npts_acf_lag, noise_length=1600, noise_model='uncorrelated')
        noise_std_sw = noise_estimator.get_noise_std()
        # cov_inv, log_cov_det = noise_estimator.calc_InversionDeterminant_cd()
        print(noise_std_sw.shape)
        # for s in range(len(stations)):
        #     for c in range(3):
        #         print(max(np.abs(data_sw[s][c].data)))

    else:
        stations = None
        data_sw = None
        greens_sw = None

    #    
    # The main computational work starts now
    #

    if comm.rank==0:
        ##
        MAXVAL = 3600
        ne, ns, nc = 6, len(stations), 3
    
        print('Evaluating surface wave misfit...\n')
        np.random.seed(1000)
        ##number of unknowns
        ndim = ne + ns + 2*ns
        nwalker = 512
        nsteps = 10000
        init = np.random.uniform(-MAXVAL, MAXVAL, (nwalker, ndim))

        print('Important parameters: ne-%d, ns-%s, nc-%d' % (ne, ns, nc))
        # ## Create the MCMC solver
        solver = MCMC_SOLVER(misfit_sw, data_sw, greens_sw, \
                          noise_std_sw, max_noise_parameter=40, M00=1.e14, method='mij_uncorrelated')
        sampler, pool = solver.get_sampler('emcee', nchains=nwalker)
        # MCMC sampling
        state = sampler.run_mcmc(init, nsteps, progress=True)
        solver.cleanup(pool)

        ## Print acceptance fraction for diagnosis
        acceptance_rate = 100 * np.mean(sampler.acceptance_fraction)
        print ('Average acceptance rate: %d' % acceptance_rate + '%')
            
        ##write the samples into files
        solver.save_chains(sampler, file_path='./', thin=2)
        
    if comm.rank==0:

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
        plot_data_greens1(event_id+'Mij_waveforms_sw_d%skm_noise_cd.png' % evdp_in_km,
            data_sw, greens_sw, process_sw, 
            misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(event_id+'Mij_beachball_sw_d%skm_noise_cd.png' % evdp_in_km,
            best_mt, stations, origin)
        
        plot_waveform_fit(best_mt.as_vector(), solver.obs, solver.greens, stations, noise_sol, tau_sol, event_id+'Waveformfit_mean.jpg', evdp_in_km)

        #
        # Plot the posterior distribution
        posterior_distribution_mij(source_type='full', flat_samples_fname=solver.chain_fname,log_prob_fname=solver.logprob_fname, thin=2, figure_fname=event_id+"_Posterior_source_parameter.jpg")
        posterior_distribution_noise(flat_samples_fname=solver.chain_fname, mt_degree=6, thin=10, stations=stations, figure_fname=event_id+'_Posterior_data_noise.jpg')
        posterior_distribution_timeshift(flat_samples_fname=solver.chain_fname, mt_degree=6, thin=10, stations=stations, figure_fname=event_id+'_Posterior_timeshift')
        print(noise_sol)
        print(tau_sol)
        print('\nFinished\n')

