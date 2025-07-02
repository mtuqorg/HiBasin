#!/usr/bin/env python

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_beachball, plot_misfit_lune, plot_data_greens2
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.grid import UnstructuredGrid
from mtuq.grid.moment_tensor import to_mt
from mtuq.util.math import to_mij, to_rho
from obspy.signal.filter import bandpass

import multiprocessing
import emcee
import sys
sys.path.insert(0, '/Users/hujy/Documents/Research/BayMTI/src/')
from likelihood import *
from utils.data_selection import data_noise_estimate_uncorrelated, get_solution
from utils.misfit_preparation import shift_greens, misfit_preparation
from utils.visualization import plot_waveform_fit

from mtuq.event import Force

if __name__=='__main__':
    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    #   


    path_data=    fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/*.[zrt]')
    path_weights= fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/weights.dat')
    path_weights_raw= fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/weights_raw.dat')
    event_id=     '20090407201255351'
    model=        'ak135'

    #
    # Surface wave measurements will be made separately
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
    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-12.,
        time_shift_max=+12.,
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
        'latitude': 61.454,
        'longitude': -149.743,
        'depth_in_m': 30000.,
        })
    evdp_in_km = int(origin.depth_in_m/1000)
   
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
        data_sw = data.map(process_sw)


        print('Reading Greens functions...\n')
        greens = download_greens_tensors(stations, origin, model, include_mt=False, include_force=True)

        print('Processing Greens functions...\n')
#         greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)

    else:
        stations = None
        data_sw = None
        greens_sw = None


    stations = comm.bcast(stations, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    greens_sw = comm.bcast(greens_sw, root=0)

    ##resample the data and greens and estimate the reference noise level
    data_sw_used, greens_sw_used, noise_std_sw= data_noise_estimate_uncorrelated(data_sw, greens_sw, sampling_rate=1)
    data_sw_array, greens_sw_array = misfit_preparation(data_sw_used, greens_sw_used)
    F00 = 1.0e11
    greens_sw_array = greens_sw_array[:, :, :, :] * F00
    
    #=========================================
    import matplotlib.pyplot as plt
    ##Simulate the new 'observatations' with a given explosive-like MT
    ns,nc,ne,nt = greens_sw_array.shape
    noise_std_sw = noise_std_sw*5 #5 iso, 8-clvd 

    sf = np.array([100,0,0]) #downward force in USE coordinate system
    syn_data = np.einsum('scet,e->sct', greens_sw_array, sf)

    np.random.seed(1000)
    fig,ax = plt.subplots(1,3, figsize=(5,3.5), sharex=True, sharey=True)
    comp_title = ['BHZ','BHR','BHT']
    max_amp = np.max(syn_data)
    for s in range(ns):
        for c in range(nc):
#             noise_sigma = np.std(syn_data[s,c], ddof=0)
#             noise_std_sw[s,c] = noise_sigma
#             print(np.std(syn_data[s,c]))
            
            noise = np.random.normal(0,1,nt)
            noise = noise/np.std(noise, ddof=0) * noise_std_sw[s,c]
            noise = bandpass(noise, freqmin=0.025, freqmax=0.0625, df=1.0)
            tmp = syn_data[s,c] + noise
            
            #plot
            ax[c].plot(tmp/max_amp+s, 'k',lw=1)
            ax[c].plot(syn_data[s,c]/max_amp+s, 'r', linestyle='dashed',lw=1.0)
            ax[c].set_xticks([0,75,150])
            ax[c].set_xlabel('Time (s)')
            ax[c].set_yticks(np.arange(ns))
            ax[c].set_yticklabels([s.station for s in stations])
            ax[c].set_title(comp_title[c])
            
            #replace the data with new syn data
            data_sw_array[s,c] = tmp
            data_sw[s].select(channel=comp_title[c])[0].data = tmp
            data_sw_used[s].select(channel=comp_title[c])[0].data = tmp 
            
        #scale up the GFs
#         for e in range(9):
# #             greens_sw_used[s][e].data *= F00 
#             greens_sw[s][e].data *= F00 ##they share the same memory
      
    plt.savefig('waveform_sf.jpg', bbox_inches='tight', dpi=300)
    plt.show()
    #=========================================
    
    #    
    # The main computational work starts now
    #
    print(greens_sw_array.shape)
    if comm.rank==0:
        print('Evaluating surface wave misfit...\n')
        ns,nc,ne,nt = greens_sw_array.shape
#         ne=3 #for SF
        np.random.seed(1000)
        ##number of unknowns
        ndim = ne + ns + 2*ns
        nwalker = 600 
        nsteps = 5000
        MAXVAL = 3600
        init = np.random.uniform(-MAXVAL, MAXVAL, (nwalker, ndim))

        print('Important parameters: ne-%d, ns-%s, nc-%d, nt-%d' % (ne, ns, nc, nt))
        ############### SAMPLING MODEL SPACE WITH EMCEE ###############
        with multiprocessing.Pool() as pool:
            log_prob_fn = log_prob_noiseamp_timeshift_force
            ## Initializa the sample
            sampler = emcee.EnsembleSampler(nwalker, ndim, log_prob_fn, args=[data_sw_array, greens_sw_array, noise_std_sw], pool=pool)
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
        mean_sol = get_solution(sampler, warm_up_steps=int(0.5*nsteps), thin=50, source_type='force')
        
        # dictionary of Mij parameters
        print("The best mt:")
        print(mean_sol.get_dict(0))
        print(mean_sol.get(0).as_dict())
        
        ##write the samples into files
        chain_dir = './'
        noise_std_sw.tofile(chain_dir + 'EMCEE_noise_std_sw_usf_d%dkm_sigma.bin' % evdp_in_km)
        flat_samples_out = sampler.get_chain(discard=0, thin=5, flat=False)
        flat_samples_out.tofile(chain_dir +'EMCEE_sw_noise_usf_d%dkm_model.bin' % evdp_in_km)
        log_prob_samples = sampler.get_log_prob(discard=0, thin=5, flat=False)
        log_prob_samples.tofile(chain_dir +'EMCEE_sw_noise_usf_d%dkm_log_prob.bin' % evdp_in_km)

        ##check the data noise and time shfits
        flat_samples = sampler.get_chain(discard=int(0.5*nsteps), thin=10, flat=True)
        noise = np.mean(MAXVAL+flat_samples[:,ne:ne+ns], axis=0) / 720 + 0.0001
        tau = np.mean(flat_samples[:,ne+ns:], axis=0) / 360
        print("noise: ", noise)
        print("tau: ", tau)

    if comm.rank==0:

        #
        # Generate figures and save results
        #

        print('Generating figures...\n')
#         best_force = mean_sol.get(0)
        force_dict = mean_sol.get_dict(0)
        
        force_tmp = UnstructuredGrid(
                dims=('F0','phi', 'h'),
                coords=(force_dict['F0']*F00, force_dict['phi'], force_dict['h']),
                callback=to_force)
        best_force = force_tmp.get(0)
        force_dict['F0'] = force_dict['F0']*F00
        
        print("best force: ", best_force.as_dict())
        print("force dict: ", force_dict)
        greens_sw = shift_greens(greens_sw, tau)
        plot_data_greens1(event_id+'USF_waveforms_sw_noise.png',
            data_sw, greens_sw, process_sw, 
            misfit_sw, stations, origin, best_force, force_dict)

#         plot_beachball(event_id+'SF_beachball_sw_noise.png',
#             best_force, stations, origin)

#         plot_waveform_fit(best_force.as_vector(), data_sw_array, greens_sw_array, stations, noise, tau, event_id+'SF_waveformfit_sw_noise_new.png')
        
        print('\nFinished\n')

