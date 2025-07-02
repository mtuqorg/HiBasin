#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1,plot_data_greens2, plot_beachball, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.grid import UnstructuredGrid
from mtuq.grid.moment_tensor import to_mt
from mtuq.util.math import to_mij, to_rho

import multiprocessing
import emcee
from likelihood import *

if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   


    path_data=    fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/*.[zrt]')
    path_weights= fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'

    #
    # Surface wave measurements will be made separately
    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='taup',
        taup_model=model,
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )
        
    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model=model,
        window_type='surface_wave',
        window_length=150.,
        capuaf_file=path_weights,
        )

    #
    # For our objective function, we will use surface wave
    # contribution only
    #
    misfit_bw = Misfit(
        norm='L2',
        time_shift_min=-2.,
        time_shift_max=+2.,
        time_shift_groups=['ZR'],
        )
        
    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-10.,
        time_shift_max=+10.,
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
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })


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
            tags=['units:m', 'type:velocity']) 


        data.sort_by_distance()
        stations = data.get_stations()

        print('Processing data...\n')
        data_bw = data.map(process_bw)
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model)

        print('Processing Greens functions...\n')
        greens.convolve(wavelet)
        greens_bw = greens.map(process_bw)
        greens_sw = greens.map(process_sw)


    else:
        stations = None
        data_bw = None
        data_sw = None
        greens_bw = None
        greens_sw = None


    stations = comm.bcast(stations, root=0)
    data_bw = comm.bcast(data_bw, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_bw = comm.bcast(greens_bw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    ##data_sw is a list of obspy data stream
    ##resample the data and greens and estimate the reference noise level
    data_bw_used = []
    data_sw_used = []
    greens_bw_used = []
    greens_sw_used = []
    for i in range(len(data_sw)):
        if len(data_sw[i]) != 0:
            data_bw[i].resample(sampling_rate=2)
            data_sw[i].resample(sampling_rate=2)
            greens_bw[i].resample(sampling_rate=2)
            greens_sw[i].resample(sampling_rate=2)
            
            data_bw_used.append(data_bw[i])
            data_sw_used.append(data_sw[i])
            greens_bw_used.append(greens_bw[i]) 
            greens_sw_used.append(greens_sw[i])    
    data_bw_used = Dataset(data_bw_used) 
    data_sw_used = Dataset(data_sw_used) 
    greens_bw_used = Dataset(greens_bw_used)
    greens_sw_used = Dataset(greens_sw_used)
    print("There are %s stations in use for this inversion" % len(data_sw_used))
    
    ns = len(data_sw_used)
    nc = 3
    components = ['Z','R','T']
    noise_std = np.ones((ns,nc)) 
    for s in range(ns):
       for c in range(3):
           noise_std[s,c] = np.std(data_sw_used[s].select(component=components[c])[0].data, ddof=0) 

    print(noise_std)
    #    
    # The main computational work starts now
    #

    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')
        ns = len(data_sw_used)
        ne = 6
        nc = 3
        nt = greens_sw_used[0][0].stats.npts
        np.random.seed(2000)
        ndim = ne 
        nwalker = 128 
        nsteps = 6000
        MAXVAL = 3600
        init = np.random.uniform(-MAXVAL, MAXVAL, (nwalker, ndim))

        print('Important parameters: ne-%d, ns-%s, nc-%d, nt-%d' % (ne, ns, nc, nt))
        ############### SAMPLING MODEL SPACE WITH EMCEE ###############
        with multiprocessing.Pool() as pool:
            log_prob_fn = log_prob2
            ## Initializa the sample
            sampler = emcee.EnsembleSampler(nwalker, ndim, log_prob_fn, args=[nt,data_bw_used,data_sw_used,greens_bw_used, greens_sw_used, misfit_bw, misfit_sw], pool=pool)
            ## Running MCMC
            state = sampler.run_mcmc(init, nsteps, progress=True)
 
        ############### POST SAMPLING ANALYSIS ###############
        ## Print acceptance fraction for diagnosis
        acceptance_rate = 100 * np.mean(sampler.acceptance_fraction)
        print ('Average acceptance rate: %d' % acceptance_rate + '%')
 
        ## Get the idea of autocorrelation in the sampled chains
        try:
            tau = sampler.get_autocorr_time(tol=0)
            autocorr_time = int(np.max(tau))
            print ('\nAutocorrelation time for each coordinates of the model space:\n    ', tau)
        except Exception as ex:
            print (ex)
            autocorr_time = int(nsteps / 50)
 
        ## Extract the chain for inspection
        warmup = 3 * autocorr_time
        thin = int(autocorr_time/2)
        flat_samples = sampler.get_chain(discard=int(0.5*nsteps), thin=10, flat=True)
        print ('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)
        print(m_sol)
        
        v = m_sol[0] / 10800
        w = m_sol[1] * np.pi / 9600             
        kappa = (m_sol[2] + MAXVAL) / 20       ##0, 360
        sigma = m_sol[3] / 40                  ##-90, 90
        h = (m_sol[4] + MAXVAL) / 7200         ##cos(dip)
        rho = to_rho((m_sol[5]+MAXVAL)/3600 + 3.5) ##Mw: 4-6

        ## moment tensor
        mean_sol = UnstructuredGrid(
            dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
            coords=(rho, v,w,kappa,sigma,h),
            callback=to_mt)
        
        # dictionary of Mij parameters
        print("The best mt:")
        print(mean_sol.get_dict(0))
        
        chain_dir = './'
        flat_samples_out = sampler.get_chain(discard=1, thin=5, flat=False)
        flat_samples_out.tofile(chain_dir +'EMCEE_run5_model.bin')
        log_prob_samples = sampler.get_log_prob(discard=1, thin=5, flat=False)
        log_prob_samples.tofile(chain_dir +'EMCEE_run5_log_prob.bin')

    if comm.rank==0:

        #
        # Generate figures and save results
        #

        print('Generating figures...\n')
        best_mt = mean_sol.get(0)
        lune_dict = mean_sol.get_dict(0)
        plot_data_greens2(event_id+'FMT_waveforms_run5.png',
            data_bw,data_sw, greens_bw, greens_sw, process_bw,process_sw, 
            misfit_bw, misfit_sw, stations, origin, best_mt, lune_dict)


        plot_beachball(event_id+'FMT_beachball_run5.png',
            best_mt, stations, origin)

        print('\nFinished\n')

