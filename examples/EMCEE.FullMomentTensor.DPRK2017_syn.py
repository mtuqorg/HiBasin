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
from hibasin.misfit.likelihood import MCMC_SOLVER
from hibasin.util.covariance_matrix import covariace_matrix
from hibasin.util.math import exponential_covariance, calc_InversionDeterminant_cd
from hibasin.util.misfit_preparation import shift_greens, shift_data,  misfit_preparation
from hibasin.visualization.plot_waveform_fit import plot_waveform_fit
from hibasin.visualization.plot_posterior import posterior_distribution_mij, posterior_distribution_noise, posterior_distribution_timeshift
from obspy.signal.filter import bandpass
from multiprocessing import shared_memory
from hibasin.misfit.misfit_preparation import to_numpy_arrays

os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method("fork", force=True)

def ned2rtp(mt_ned):
    Mxx,Myy,Mzz, Mxy,Mxz,Myz = mt_ned
    Mrr=Mzz
    Mtt=Mxx
    Mpp=Myy
    Mrt=Mxz
    Mrp=-Myz
    Mtp=-Mxy
    return np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])

if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   

    path_data=    fullpath('data/20170903033001000/*.BH[ZRT].sac')
    path_weights= fullpath('data/20170903033001000/weights_surf.dat')
    event_id=     '20170903033001000'
    model=        'mdj3'

    #
    # Surface wave measurements will be made separately
    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.02,
        freq_max=0.05,
        pick_type='CPS_metadata',
        CPS_database='data/grn_2017_2d/',
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
        time_shift_min=-10.,
        time_shift_max=+10.,
        time_shift_groups=['ZR','T']
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
        magnitude=4.9)

    #
    # Origin time and location will be fixed. For an example in which they 
    # vary, see examples/GridSearch.DoubleCouple+Magnitude+Depth.py
    #
    # See also Dataset.get_origins(), which attempts to create Origin objects
    # from waveform metadata
    #

    origin = Origin({
        'time': '2017-09-03T03:30:01.760000Z',
        'latitude': 41.3,
        'longitude': 129.078,
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

        print('Reading Greens functions...\n')
        db = open_db('data/grn_2017_2d/mdj3',  format='CPS', model=model)
        greens = db.get_greens_tensors(stations, origin)

        print('Processing Greens functions...\n')
#         greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)

        ##resample t and greens
        # for s in range(len(stations)):
        #     data_sw[s].resample(1)
        #     greens_sw[s].resample(1)
        #     #update the delta in dataset.station
        #     data_sw[s].station._refresh('delta',1)
        #     greens_sw[s].station._refresh('delta',1)

        stations = comm.bcast(stations, root=0)
        data_sw = comm.bcast(data_sw, root=0)
        greens_sw = comm.bcast(greens_sw, root=0)

        ##estimate the noise strength and covariance matrix
        data_noise = read(path_data, format='sac', 
                event_id=event_id,
                station_id_list=station_id_list,
                tags=['units:m', 'type:displacement'])
        # for traces in data_noise:
        #     traces.resample(1)
        npts_acf_lag = data_sw[0][0].stats.npts
        noise_estimator = covariace_matrix(origin, data_noise, npts_acf_lag, noise_length=3000, noise_model='uncorrelated')
        noise_std_sw = noise_estimator.get_noise_std()
        # cov_inv, log_cov_det = noise_estimator.calc_InversionDeterminant_cd()
        print(noise_std_sw.shape)

    else:
        stations = None
        data_sw = None
        greens_sw = None

###synthetic data
    import matplotlib.pyplot as plt
    ##Simulate the new 'observatations' with a given explosive-like MT
    data_sw_array, greens_sw_array = to_numpy_arrays(data_sw, greens_sw)
    ns,nc,ne,nt = greens_sw_array.shape
    print(greens_sw_array.shape)
    np.save('obs_2017', data_sw_array)
    np.save('gf_2017', greens_sw_array)

    #ISO source
    mt = ned2rtp(np.loadtxt('../tests/mt_input.txt'))
    syn_data = np.einsum('scet,e->sct', greens_sw_array, mt)
        
    fig,ax = plt.subplots(1,3, figsize=(4.5,7), sharex=True, sharey=True)
    comp_title = ['BHZ','BHR','BHT']
    max_amp = 1.5*np.max(syn_data)
    np.random.seed(4000)
    for s in range(ns):
        for c in range(nc):

            #generate a correlated data noise series based on the mean and covariance matrix
            noise = np.random.normal(0, 1, nt)
            noise = bandpass(noise, freqmin=0.02, freqmax=0.05, df=1.0) 
            taper(noise, taper_fraction=0.2)
            d_rms = np.sqrt(np.mean(np.square(noise)))
            noise = noise/d_rms * noise_std_sw[s,c]
            # print('sigma of noise: ', np.std(noise, ddof=0))
            tmp = syn_data[s,c] + noise
            
            #plot
            ax[c].plot(tmp/max_amp+s, 'k',lw=1)
            ax[c].plot(syn_data[s,c]/max_amp+s, 'r', linestyle='dashed',lw=1)
            ax[c].plot(noise/max_amp+s-0.3, 'blue',lw=.5)
            ax[c].set_xticks([0,75,150])
            ax[c].set_xlabel('Time (s)')
            ax[c].set_yticks(np.arange(ns))
            ax[c].set_yticklabels([s.network + '.' + s.station for s in stations])
            ax[c].set_title(comp_title[c])
            
            #replace the data with new syn data
            data_sw[s].select(channel=comp_title[c])[0].data = tmp
            data_sw_array[s,c] = tmp
    plt.savefig('waveform.jpg', bbox_inches='tight', dpi=300)
    plt.close()

    ##calculate the upper bound of VR
    res = data_sw_array - syn_data
    vr = (1 - np.sum(res**2)/np.sum(data_sw_array**2)) * 100
    print("The upper bound of VR could be %.1f%%" % vr)
####

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
        ndim = ne + ns + len(misfit_sw.time_shift_groups)*ns
        nwalker = 512
        nsteps = 10000
        init = np.random.uniform(-MAXVAL, MAXVAL, (nwalker, ndim))

        print('Important parameters: ne-%d, ns-%s, nc-%d' % (ne, ns, nc))
        # ## Create the MCMC solver
        solver = MCMC_SOLVER(misfit_sw, data_sw, greens_sw, \
                          noise_std_sw, max_noise_parameter=400, M00=5.0e14, method='mij_uncorrelated')
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
        plot_data_greens1(event_id+'_Mij_waveforms_sw_d%skm_noise_cd.png' % evdp_in_km,
            data_sw, greens_sw, process_sw, 
            misfit_sw, stations, origin, best_mt, lune_dict)

        plot_beachball(event_id+'_Mij_beachball_sw_d%skm_noise_cd.png' % evdp_in_km,
            best_mt, stations, origin)
        
        plot_waveform_fit(best_mt.as_vector(), solver.obs, solver.greens, stations, noise_sol, tau_sol, event_id+'_Waveformfit_mean.jpg', evdp_in_km)

        #
        # Plot the posterior distribution
        posterior_distribution_mij(source_type='full', flat_samples_fname=solver.chain_fname,log_prob_fname=solver.logprob_fname, thin=10, ratio=0.5, figure_fname=event_id+"_Posterior_source_parameter.jpg")
        posterior_distribution_noise(flat_samples_fname=solver.chain_fname, mt_degree=6, thin=10, ratio=0.5,stations=stations, figure_fname=event_id+'_Posterior_data_noise.jpg')
        posterior_distribution_timeshift(solver, mt_degree=6, thin=10, ratio=0.5,stations=stations, figure_fname=event_id+'_Posterior_timeshift.jpg')
        print(noise_sol)
        print(tau_sol)
        print('\nFinished\n')

