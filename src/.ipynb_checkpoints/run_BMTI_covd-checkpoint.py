import corner
import emcee
import emcee
import os
import multiprocessing
import matplotlib
import numpy as np
from netCDF4 import Dataset
import pyrocko.moment_tensor as mtm
from numpy.linalg import inv, det, slogdet
from scipy.linalg import cholesky, solve_triangular
from numpy.fft import rfft, irfft, rfftfreq
from matplotlib import pyplot as plt

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens2, plot_beachball, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid

MAXVAL = 2000
rad = np.pi / 180

if __name__ == '__main__':

    #
    # Carries out grid search over all moment tensor parameters
    #
    # USAGE
    #   mpirun -n <NPROC> python GridSearch.FullMomentTensor.py
    # 
    
    path_data=    fullpath('data/20090407201255351/*.[zrt]')
    path_weights= fullpath('data/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'
    

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
    # User-supplied weights control how much each station contributes to the
    # objective function
    #

    station_id_list = parse_station_codes(path_weights)

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })
    
    wavelet = Trapezoid(magnitude=4.5)

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
        greens.convolve(wavelet)
        greens_sw = greens.map(process_sw)


    else:
        stations = None
        data_sw = None
        greens_sw = None


    stations = comm.bcast(stations, root=0)
    data_sw = comm.bcast(data_sw, root=0)
    ## mtuq uses an up-south-east basis convention
    greens_sw = comm.bcast(greens_sw, root=0)

    misfit_sw = Misfit(
        norm='L2',
        time_shift_min=-3.,
        time_shift_max=+3.,
        time_shift_groups=['ZR','T'],
        )
    
    #
    # The main computational work starts now
    #
    #define the dimension of problem
    ns = len(data_sw) #The number of station
    ne = 6            #The number of MT parameters
    nc = 3            #The number of components
    nt = greens_sw[0].stats.npts 
    delta = greens_sw[0].stats.delta
    
    np.random.seed(1000)
    ## Problem size definition and initialisation
    ndim = ne + 3*ns # number of MT params and number of stations for noise amplitude factors
    nwalker = 512
    nsteps = 10000
    init = np.random.uniform(-MAXVAL, MAXVAL, (nwalker, ndim))

    print('Important parameters: ne-%d, ns-%s, nc-%d, nt-%d' % (ne, ns, nc, nt))
    ############### SAMPLING MODEL SPACE WITH EMCEE ###############
    with multiprocessing.Pool() as pool:
        log_prob_fn = log_prob_noiseamp_timeshift
        ## Initializa the sample
        sampler = emcee.EnsembleSampler(nwalker, ndim, log_prob_fn, args=[data_sw, greens_sw, misfit_sw, noise_std, delta], pool=pool)
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
        print ('\nAutocorrelation time for each coordinates of the model space:\n', tau)
    except Exception as ex:
        print (ex)
        autocorr_time = int(nsteps / 50)

    ## Extract the chain for inspection
    warmup = 3 * autocorr_time
    thin = int(autocorr_time/2)
    flat_samples = sampler.get_chain(discard=warmup, thin=thin, flat=True)
    print ('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
    
    #

    model_samples = sampler.get_chain(discard=1, thin=5, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=1, thin=5, flat=True)

    ##mean solution
    flat_samples = sampler.get_chain(discard=5000, thin=thin, flat=True)
    print(np.mean(flat_samples[:,:6], axis=0))

    ################
    ##visualization by calling the functions from mtuq

