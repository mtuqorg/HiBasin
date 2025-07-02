import sys
sys.path.insert(0,'/Users/hujy/Documents/Research/BayMTI/baymti/')
from utils.visualization import plot_data_covariance_matrix, posterior_distribution, posterior_distribution_noise, posterior_distribution_timeshift
import numpy as np
from mtuq.util.cap import parse_station_codes
from mtuq.util import fullpath

dirname = '/Users/hujy/Documents/Research/BayMTI/src/'
fname = 'EMCEE_sw_noise_dev_log_prob.bin'

mt_degree = 5 #4 for dc, 5 for deviatoric, 6 for full mt
ns = 8   #no of stations
nchains = 600
log_prob = np.fromfile(dirname + fname)
num = int(len(log_prob)/nchains)
log_prob = log_prob.reshape((num, nchains))
samples = np.fromfile(dirname + 'EMCEE_sw_noise_dev_model.bin')#'EMCEE_sw_noise_fmt_model.bin')
num_pars = mt_degree+ns+ns+ns
samples = samples.reshape((num, nchains, num_pars))

##flatten the samples
samples = samples.reshape(-1, samples.shape[-1])
log_prob = log_prob.reshape(-1)
print(log_prob.shape)
print(samples.shape)
posterior_distribution(source_type='deviatoric', flat_samples=samples, log_prob=log_prob, thin=50, figure_fname='Dev_posterior.jpg')


path_data=    fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/*.[zrt]')
path_weights= fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/weights.dat')
event_id=     '20090407201255351'
model=        'ak135'

stations = parse_station_codes(path_weights)
print(stations)
sigma = np.fromfile('noise_std_sw_sigma.bin').reshape(8,3)
npts = 150 
k = np.mean((samples[:,mt_degree:mt_degree+ns]+ 3600)/720 + 0.0001, axis=0) 
plot_data_covariance_matrix(sigma, stations,npts,'./Noise_cd_dev.png')
plot_data_covariance_matrix(np.einsum('sc,s->sc',sigma,k), stations,npts,'./Noise_cd_recovered_dev.png')

posterior_distribution_noise(flat_samples=samples, mt_degree=mt_degree, ns=ns, thin=50, stations=stations, figure_fname='dev_noise.jpg')
posterior_distribution_timeshift(flat_samples=samples, mt_degree=mt_degree, ns=ns, thin=50, stations=stations, figure_fname='dev_tau_R.jpg', wave_type='R')
posterior_distribution_timeshift(flat_samples=samples, mt_degree=mt_degree, ns=ns, thin=50, stations=stations, figure_fname='dev_tau_L.jpg', wave_type='L')

