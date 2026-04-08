import sys
sys.path.insert(0,'/Users/hujy/Documents/Research/BayMTI/src/')
from utils.visualization import plot_data_covariance_matrix, posterior_distribution, posterior_distribution_noise, posterior_distribution_timeshift
from utils.math import exponential_covariance
import numpy as np
from mtuq.util.cap import parse_station_codes
from mtuq.util import fullpath

evdp = 30
dirname = '/Users/hujy/Documents/Research/BayMTI/examples/'
fname = 'EMCEE_sw_noise_tt2015_syn_d%skm_log_prob_cd.bin' % evdp

mt_degree = 6 #4 for dc, 5 for deviatoric, 6 for full mt, 3 for force
ns = 8   #no of stations
nchains = 600

log_prob = np.fromfile(dirname + fname)
#num = int(len(log_prob)/nchains)
#log_prob = log_prob.reshape((num, nchains))
samples = np.fromfile(dirname + 'EMCEE_sw_noise_tt2015_syn_d%skm_model_cd.bin' % evdp)#'EMCEE_sw_noise_fmt_model.bin')
num_pars = mt_degree+ns+ns+ns
#samples = samples.reshape((num, nchains, num_pars))

##flatten the samples
num = int(len(log_prob/num_pars))
samples = samples.reshape(num, num_pars)
log_prob = log_prob.reshape(-1)
#samples = samples.reshape(-1, samples.shape[-1])
#log_prob = log_prob.reshape(-1)
print(log_prob.shape)
print(samples.shape)
print(samples[-1])
posterior_distribution(source_type='full', flat_samples=samples, log_prob=log_prob, thin=1, figure_fname='FMT_tt2015_rev_syn_d%skm_posterior.jpg' % evdp)


path_data=    fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/*.[zrt]')
path_weights= fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/weights.dat')
event_id=     '20090407201255351'
model=        'ak135'

stations = parse_station_codes(path_weights)
print(stations)
sigma = np.fromfile('noise_std_sw_sigma.bin').reshape(8,3)
npts = 150 
half_num = int(0.5*num)
k = 2*np.mean((samples[half_num:,mt_degree:mt_degree+ns]+ 3600)/720 + 0.0001, axis=0) 
# plot_data_covariance_matrix(sigma, stations,npts,'./Noise_cd_tt2015_d%skm_syn.png' % evdp)
# plot_data_covariance_matrix(np.einsum('sc,s->sc',sigma,k), stations,npts,'./Noise_cd_recovered_fmt_tt2015_syn_d%skm.png' % evdp)

#
##plot the reference covariance matrix
#generate covariance matrix for exp decay noise model
nt = 150
nc = 3

cov_matrix = exponential_covariance(nt,2)
cov_d = np.zeros((ns,nc,nt,nt))
for s in range(ns):
    for c in range(nc):
        cov_d[s,c] = cov_matrix
plot_data_covariance_matrix(np.einsum('sc,s->sc',sigma,k), stations,npts,'./Noise_cd_tt2015_d%skm_syn_recovered_expCd.png' % evdp, cov_d)
# plot_data_covariance_matrix(np.einsum('sc,s->sc',sigma,k), stations,npts,'./Noise_cd_tt2015_d%skm_syn_expCd_difference.png' % evdp, cov_d)

samples[:, mt_degree:mt_degree+ns]  *= 2
posterior_distribution_noise(flat_samples=samples, mt_degree=mt_degree, ns=ns, thin=50, stations=stations, figure_fname='TT2015_fmt_syn_d%skm_noise.jpg' % evdp)
posterior_distribution_timeshift(flat_samples=samples, mt_degree=mt_degree, ns=ns, thin=50, stations=stations, figure_fname='TT2015_fmt_syn_d%skm_tau_R.jpg' % evdp, wave_type='R')
posterior_distribution_timeshift(flat_samples=samples, mt_degree=mt_degree, ns=ns, thin=50, stations=stations, figure_fname='TT2015_fmt_syn_d%skm_tau_L.jpg' % evdp, wave_type='L')


