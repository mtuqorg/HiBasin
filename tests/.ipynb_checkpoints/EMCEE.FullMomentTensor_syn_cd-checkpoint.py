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
from src.misfit.likelihood import Loglikelihood, MCMC_FMT
from src.util.math import exponential_covariance, calc_InversionDeterminant_cd
from src.util.data_selection import data_noise_estimate_uncorrelated, get_solution
from src.util.misfit_preparation import shift_greens, misfit_preparation
from src.visualization.visualization import plot_waveform_fit
from obspy.signal.filter import bandpass
from multiprocessing import shared_memory

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

    path_data=    fullpath('/Users/u7091895/Documents/Research/BayMTI/HiBaysin/data/20090407201255351/*.[zrt]')
    path_weights= fullpath('/Users/u7091895/Documents/Research/BayMTI/HiBaysin/data/20090407201255351/weights.dat')
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
        greens = download_greens_tensors(stations, origin, model)

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
    
  
    # ##calculate the inverse of covariance matrix, cov_d, for pre-event ambient noise series
    # cov_inv, log_cov_det = calc_InversionDeterminant_cd(cov_d)
    #=========================================
    import matplotlib.pyplot as plt
    ##Simulate the new 'observatations' with a given explosive-like MT
    ns,nc,ne,nt = greens_sw_array.shape
    noise_std_sw = noise_std_sw * 1.5#2.5 CLVD #5 for 1km#5 iso, 8-clvd. 3 for 30km, 1.5 dc
    noise_std_sw.tofile('noise_std_sw_sigma.bin')
    # noise_std_sw = np.ones((ns,nc)) * 1.0e-6
    #ISO source
    # mt = ned2rtp(np.loadtxt('mt_input.txt'))
    # DC source
    # mt = np.array([-8.88783183e+15,  4.66977228e+16, -3.78098910e+16,  \
    #                3.71126807e+15, 2.09101333e+16,  1.47054371e+16])
    mt = np.array([-0.089, 0.467, -0.378,  \
                0.037,  0.209, 0.147]) *1.0e17
    #CLVD source
    # mt = ned2rtp(np.array([2.760, -1.661, -1.099, 0.707, 1.048, 0.634])*3.0e16)
    
    syn_data = np.einsum('scet,e->sct', greens_sw_array, mt)
    
    #generate covariance matrix for exp decay noise model
    cov_matrix = exponential_covariance(nt,4)
    mean = np.zeros(nt)
    
    fig,ax = plt.subplots(1,3, figsize=(4.5,7), sharex=True, sharey=True)
    comp_title = ['BHZ','BHR','BHT']
    max_amp = 1.5*np.max(syn_data)
    np.random.seed(4000)
    for s in range(ns):
        for c in range(nc):

            #generate a correlated data noise series based on the mean and covariance matrix
            noise = np.random.multivariate_normal(mean, cov_matrix) 
            noise = bandpass(noise, freqmin=0.025, freqmax=0.0625, df=1.0) 
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
            data_sw_array[s,c] = tmp
            data_sw[s].select(channel=comp_title[c])[0].data = tmp
            data_sw_used[s].select(channel=comp_title[c])[0].data = tmp   
    plt.savefig('waveform.jpg', bbox_inches='tight', dpi=300)
    plt.close()

    ##calculate the upper bound of VR
    res = data_sw_array - syn_data
    vr = (1 - np.sum(res**2)/np.sum(data_sw_array**2)) * 100
    print("The upper bound of VR could be %.1f%%" % vr)
    
    ##Scale up for Mij
    M00 = 1.0e15
    greens_sw_array = greens_sw_array[:, :, :, :] * M00
    #=========================================

##test shared memory
    cov_matrix = exponential_covariance(150, 4)
    cov_d = np.broadcast_to(cov_matrix, (ns, nc, nt, nt))
    cov_inv, log_cov_det = calc_InversionDeterminant_cd(cov_d)

    #    
    # The main computational work starts now
    #

    if comm.rank==0:
        ##
        MAXVAL = 3600
        ns,nc,ne,nt = greens_sw_array.shape
        emcee_dataset = {
           'MAXVAL': MAXVAL,
           'delta': 1.0,
           'obs': data_sw_array,
           'noise_std': noise_std_sw,
           'green_tensor': greens_sw_array,
           'cov_inv': cov_inv,
           'log_cov_det': log_cov_det,
        }
        # log_prob_fn = Loglikelihood(emcee_dataset, 'full_mij_correlated_exp')
  
        print('Evaluating surface wave misfit...\n')
        np.random.seed(2000)
        ##number of unknowns
        ndim = ne + ns + 2*ns
        nwalker = 600
        nsteps = 10000
        init = np.random.uniform(-MAXVAL, MAXVAL, (nwalker, ndim))

        print('Important parameters: ne-%d, ns-%s, nc-%d, nt-%d' % (ne, ns, nc, nt))
        ############### SAMPLING MODEL SPACE WITH EMCEE ###############
        # with mp.Pool() as pool:
        #     ## Initializa the sample
        #     sampler = emcee.EnsembleSampler(nwalker, ndim, log_prob_fn, pool=pool)
        #     ## Running MCMC
        #     state = sampler.run_mcmc(init, nsteps, progress=True)

        solver = MCMC_FMT(emcee_dataset, 'mij_correlated_exp')
        sampler, pool = solver.get_sampler(nchains=nwalker)
        state = sampler.run_mcmc(init, nsteps, progress=True)
        pool.close()
        pool.join()
        solver.cleanup()
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
        mean_sol = get_solution(sampler, warm_up_steps=int(0.5*nsteps), thin=100, source_type='mij')
        
        # dictionary of Mij parameters
        print("The best mt:")
        print(mean_sol.get_dict(0))
        
        ##write the samples into files
        chain_dir = './'
        flat_samples_out = sampler.get_chain(discard=0, thin=5, flat=False)
        flat_samples_out.tofile(chain_dir +'EMCEE_sw_noise_mij_syn_d%skm_model_cd.bin' % evdp_in_km)
        log_prob_samples = sampler.get_log_prob(discard=0, thin=5, flat=False)
        log_prob_samples.tofile(chain_dir +'EMCEE_sw_noise_mij_syn_d%skm_log_prob_cd.bin' % evdp_in_km)
        
        ##check the data noise and time shfits
        flat_samples = sampler.get_chain(discard=int(0.5*nsteps), thin=200, flat=True)
        noise = np.mean(MAXVAL+flat_samples[:,ne:ne+ns], axis=0) / 720 + 0.0001
        tau = np.mean(flat_samples[:,ne+ns:], axis=0) / 360
 
        print("noise: ", noise)
        print("tau: ", tau)

    if comm.rank==0:

        #
        # Generate figures and save results
        #

        print('Generating figures...\n')
        best_mt = mean_sol.get(0)
        lune_dict = mean_sol.get_dict(0)
        greens_sw = shift_greens(greens_sw, tau)
        plot_data_greens1(event_id+'Mij_waveforms_sw_syn_d%skm_noise_cd.png' % evdp_in_km,
            data_sw, greens_sw, process_sw, 
            misfit_sw, stations, origin, best_mt, lune_dict)


        plot_beachball(event_id+'Mij_beachball_sw_syn_d%skm_noise_cd.png' % evdp_in_km,
            best_mt, stations, origin)
        
        plot_waveform_fit(best_mt.as_vector(), data_sw_array, greens_sw_array/M00, stations, noise, tau, event_id+'Mij_waveformfit_sw_syn_d%skm_noise_cd.png' % evdp_in_km, evdp_in_km)

        print('\nFinished\n')

