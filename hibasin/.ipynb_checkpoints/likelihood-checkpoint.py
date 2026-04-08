import numpy as np
from netCDF4 import Dataset
from numpy.fft import fft, ifft, fftfreq
from mtuq.grid import UnstructuredGrid
from mtuq.util.math import to_mij, to_rho, to_rtp
from mtuq.grid.force import to_force
from mtuq.grid.moment_tensor import to_mt
from mtuq.event import MomentTensor
from mtuq.grid import UnstructuredGrid
from mtuq import Dataset
import time
import sys
sys.path.insert(0, '/Users/hujy/Documents/Research/BayMTI/src/')
from utils.math import to_lune, Tashiro2MT6, ned2rtp, numerical_jacobian,to_mij_rev
from utils.math import calc_InversionDeterminant_cd

##Jacobian determinant
# import jax.numpy as jnp
# from jax import jacfwd
from scipy.optimize import approx_fprime

MAXVAL = 3600

## Definition of prior probability
def log_prior(m):
    '''
    m contains parameters of the MT source, time shifts, and data noise
    '''
    if np.any(m < -MAXVAL) or np.any(m > MAXVAL):
        return -np.inf
    else:
        return 0

## Definition of log posterior probability
def log_prob(m, nt, data_sw, greens_sw, misfit_sw):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    ns = len(data_sw)
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 3.5)   ##Mw: 3.5-5.5
 
    ## moment tensor
    source = UnstructuredGrid(
        dims=('v', 'w', 'kappa', 'sigma', 'h', 'rho'),
        coords=(v,w,kappa,sigma,h,rho),
        callback=to_mt) 

    res_sw = misfit_sw(data_sw, greens_sw, source)
    
    noise_amp = 0.05
    lp1 = res_sw / 0.0025
    lp2 = np.sum(nt * 2 * ns*3*np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

def log_prob2(m, nt, data_bw, data_sw, greens_bw, greens_sw, misfit_bw, misfit_sw):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    ns = len(data_sw)
    ##convert to lune paramters
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 3.5)   ##Mw: 3.5-5.5
 
    ## moment tensor
    source = UnstructuredGrid(
        dims=('v', 'w', 'kappa', 'sigma', 'h', 'rho'),
        coords=(v,w,kappa,sigma,h,rho),
        callback=to_mt) 

    res_sw = misfit_sw(data_bw, greens_bw, source) + misfit_sw(data_sw, greens_sw, source)
    
    noise_amp = 0.05
    lp1 = res_sw / 0.0025
    lp2 = np.sum(nt * 2 * ns*3*np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

def log_prob_noiseamp_mtuq(m, nt, data_sw, greens_sw, noise_std, misfit_sw):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    ns = len(data_sw)
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 3.5)   ##Mw: 3.5-5.5
 
    ## moment tensor
    source = UnstructuredGrid(
        dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho,v,w,kappa,sigma,h),
        callback=to_mt) 
    ## noise amplitude
    amp = (MAXVAL+m[6:(6+ns)])/720

    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    ## 
    for s in range(ns):
        for e in range(4):
            #Z component
            greens_sw[s][e].data /= noise_amp[s,0] 
            #R component
            greens_sw[s][e+4].data /= noise_amp[s,1] 
        #T component
        for e in range(2):
            greens_sw[s][e+8].data /= noise_amp[s,2] 

        for c,comp in enumerate(['Z','R','T']): 
            data_sw[s].select(component=comp)[0].data /= noise_amp[s,c]

    #Debug
    #res_sw = misfit_sw(Dataset([data_sw[s]]), Dataset([greens_sw[s]]), source)
    res_sw = misfit_sw(data_sw, greens_sw, source)
    
    lp1 = res_sw 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

def log_prob_noiseamp(m, nt, data_sw, greens_sw, noise_std, misfit_sw):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    ns = len(data_sw)
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 3.5)   ##Mw: 3.5-5.5
 
    ## moment tensor
    source = UnstructuredGrid(
        dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho,v,w,kappa,sigma,h),
        callback=to_mt) 
    ## noise amplitude
    amp = (MAXVAL+m[6:(6+ns)])/720

    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    ## 
    for s in range(ns):
        for e in range(4):
            #Z component
            greens_sw[s][e].data /= noise_amp[s,0] 
            #R component
            greens_sw[s][e+4].data /= noise_amp[s,1] 
        #T component
        for e in range(2):
            greens_sw[s][e+8].data /= noise_amp[s,2] 

        for c,comp in enumerate(['Z','R','T']): 
            data_sw[s].select(component=comp)[0].data /= noise_amp[s,c]

    #Debug
    #res_sw = misfit_sw(Dataset([data_sw[s]]), Dataset([greens_sw[s]]), source)
    res_sw = misfit_sw(data_sw, greens_sw, source)
    
    lp1 = res_sw 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

def log_prob_noiseamp_timeshift_DC(m, data, greens, noise_std):
    '''
    Inversion for DC sources with treatment of time shifts as free parameters using TT2015 parameterization
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,nt = data.shape
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = 0
    w = 0           
    kappa = (m[0] + MAXVAL) / 20             ##0, 360
    sigma = m[1] / 40                        ##-90, 90
    h = (m[2] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[3]+MAXVAL)/3600 + 4)   ##Mw: 4-6
 
    mij = to_mij(rho,v,w,kappa,sigma,h)
    ## noise amplitude
    amp = (m[4:(4+ns)]+MAXVAL)/720 + 0.0001
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[4+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2)

def log_prob_noiseamp_timeshift_Deviatoricmt(m, data, greens, noise_std):
    '''
    Inversion for deviatoric MTs with treatment of time shifts as free parameters using TT2015 parameterization
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,nt = data.shape
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = 0           
    kappa = (m[1] + MAXVAL) / 20             ##0, 360
    sigma = m[2] / 40                        ##-90, 90
    h = (m[3] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[4]+MAXVAL)/3600 + 4)   ##Mw: 4-6
 
    mij = to_mij(rho,v,w,kappa,sigma,h)
    ## noise amplitude
    amp = (m[5:(5+ns)]+MAXVAL)/720 + 0.0001
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[5+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2)

def log_prob_noiseamp_timeshift_Fullmt(m, data, greens, noise_std):
    '''
    Full MT Inversion with treatment of time shifts as free parameters using TT2015 parameterization
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,nt = data.shape
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 4)     ##Mw: 4-6 for small-to-moderate events
 
    mij = to_mij(rho,v,w,kappa,sigma,h) #in up-south-east convention
    ## noise amplitude
    amp = (m[6:(6+ns)]+MAXVAL)/720 + 0.0001 #[0,10]
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[6+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2)

def log_prob_noiseamp_timeshift_Fullmt_correction(m, data, greens, noise_std):
    '''
    Full MT Inversion with treatment of time shifts as free parameters using TT2015 parameterization.
    Test the Jacobian correction for the transformation from TT2015 to mij for shallow events
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,nt = data.shape
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    mij = to_mij_rev(m[:6]) #in up-south-east convention
    # mij = to_mij(rho,v,w,kappa,sigma,h)
    ## noise amplitude
    amp = (m[6:(6+ns)]+MAXVAL)/720 + 0.0001 #[0,10]
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[6+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))

    #calculate the Jacobian determinant of the transformation from TT2015 to Mij.
    J = numerical_jacobian(to_mij_rev, m[:6])
    det_J = np.linalg.det(J)
    
    # jacobian_fn = jacfwd(to_mij_rev)
    # J = jacobian_fn(x)
    # det_J = jnp.linalg.det(J)

    # eps = np.sqrt(np.finfo(float).eps)  # Small step size for finite differences
    # J = np.array([approx_fprime(x, lambda x_i: to_mij_rev(x_i)[i], eps) for i in range(len(x))])
    # # Compute determinant
    # det_J = np.linalg.det(J)
    
    ## return
    return -0.5 * (lp1 + lp2) + np.log(np.abs(det_J))

def log_prob_noiseamp_timeshift_force(m, data, greens, noise_std):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,nt = data.shape
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    phi = (m[0]+ MAXVAL) / 20   #[0, 360]
    h = m[1] / 3600            #[-1,1]
    F0 = m[2] + MAXVAL      #[0, 7200]
 
    fij = to_rtp(F0, phi, h)
    ## noise amplitude
    amp = (m[3:(3+ns)]+MAXVAL)/720 + 0.0001
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[3+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, fij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2)

def log_prob_noiseamp_timeshift_mtsf(m, data, greens, noise_std):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,nt = data.shape
    ##convert to MT lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 3.5)   ##Mw: 3.5-5.5
    mij = to_mij(rho,v,w,kappa,sigma,h)
    
    #force
    phi = (m[6]+ MAXVAL) / 20   #[0, 360]
    h = m[7] / 3600             #[-1,1]
    F0 = m[8] + MAXVAL          #[0, 7200]
    fij = to_rtp(F0, phi, h)
    
    ## noise amplitude
    amp = (m[9:(9+ns)]+MAXVAL)/720 + 0.0001 #[0,10]
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[9+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, np.concatenate((mij,fij)))
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2)

def log_prob_noiseamp_timeshift_mij(m, data, greens, noise_std):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,ne,nt = greens.shape
    ##use the primitive MT in Bayesian sampling
    mij = m[:ne] #up-south-east convention

    ## noise amplitude
    amp = (m[ne:(ne+ns)]+MAXVAL)/720 + 0.0001
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[ne+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

def log_prob_noiseamp_timeshift_TashiroMT(m, data, greens, noise_std):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,ne,nt = greens.shape
    ##use the primitive MT in Bayesian sampling
    m[:5] = (MAXVAL + m[:5]) / 7200 #x_i (1-5) ~ (0,1)
    m[5] = (m[5]+MAXVAL)/3600 + 4      #Mw 4-6
    mij = Tashiro2MT6(m[:6]) 
    mij = ned2rtp(mij) #up-south-east convention
    
    ## noise amplitude
    amp = (m[ne:(ne+ns)]+MAXVAL)/720 + 0.0001
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[6+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res = np.einsum('sct,sc->sct', data-pred_shifted, 1/noise_amp)
    
    lp1 = np.sum(res*res) 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

def log_prob_noiseamp2(m, nt, data_bw, data_sw, greens_bw, greens_sw, noise_std, misfit_bw, misfit_sw):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    ns = len(data_sw)
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 3.5)   ##Mw: 3.5-5.5
 
    ## moment tensor
    source = UnstructuredGrid(
        dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho,v,w,kappa,sigma,h),
        callback=to_mt) 
    ## noise amplitude
    amp = (MAXVAL+m[6:(6+ns)])/720

    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    ## 
    for s in range(ns):
        for e in range(4):
            #Z component
            greens_sw[s][e].data /= noise_amp[s,0] 
            #R component
            greens_sw[s][e+4].data /= noise_amp[s,1] 
        #T component
        for e in range(2):
            greens_sw[s][e+8].data /= noise_amp[s,2] 

        for c,comp in enumerate(['Z','R','T']): 
            data_sw[s].select(component=comp)[0].data /= noise_amp[s,c]

    #res_sw = misfit_sw(Dataset([data_sw[s]]), Dataset([greens_sw[s]]), source)
    res_sw = misfit_sw(data_sw, greens_sw, source)
    
    lp1 = res_sw 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

def log_prob_noiseamp_v2(m, nt, data_sw, greens_sw, noise_std, misfit_sw):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    ns = len(data_sw)
    ##convert to lune paramters
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 3.5)   ##Mw: 3.5-5.5
 
    ## moment tensor
    source = UnstructuredGrid(
        dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
        coords=(rho,v,w,kappa,sigma,h),
        callback=to_mt) 
    ## noise amplitude
    amp = (MAXVAL+m[6:(6+ns)])/720

    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    noise_sigma = np.ones((ns,10))
    noise_sigma[:,:4] *= noise_amp[:,0]
    noise_sigma[:,4:8] *= noise_amp[:,1]
    noise_sigma[:,8:10] *= noise_amp[:,2]
    
    ## 
    for s in range(ns):
        for e in range(4):
            #Z component
            greens_sw[s][e].data /= noise_amp[s,0] 
            #R component
            greens_sw[s][e+4].data /= noise_amp[s,1] 
        #T component
        for e in range(2):
            greens_sw[s][e+8].data /= noise_amp[s,2] 

        for c,comp in enumerate(['Z','R','T']): 
            data_sw[s].select(component=comp)[0].data /= noise_amp[s,c]

    #res_sw = misfit_sw(Dataset([data_sw[s]]), Dataset([greens_sw[s]]), source)
    res_sw = misfit_sw(data_sw, greens_sw, source)
    
    lp1 = res_sw 
    lp2 = np.sum(nt * 2 * np.log(noise_amp))
    ## return
    return -0.5 * (lp1 + lp2) 

###========================================================================###
### Define log posterior probability for the cases of correlated data noise
###========================================================================###
#generate covariance matrix for exp decay noise model
#for debug
from utils.math import exponential_covariance, calc_InversionDeterminant_cd, calcInversionDeterminant
ne,ns,nc, nt = 6,8,3,150
cov_matrix = exponential_covariance(nt,4)

cov_d = np.zeros((ns,nc,nt,nt))
for s in range(ns):
    for c in range(nc):
        cov_d[s,c] = cov_matrix
##calculate the inverse of covariance matrix, cov_d, for pre-event ambient noise series
cov_inv, log_cov_det = calc_InversionDeterminant_cd(cov_d)
##
#

def log_prob_noisecov_timeshift_Fullmt(m, data, greens, noise_std):
    '''
    Full MT Inversion with treatment of time shifts as free parameters using TT2015 parameterization
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,nt = data.shape
    ##convert to lune paramters as transformed parameters in Bayesian sampling
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 4)     ##Mw: 4-6 for small-to-moderate events
 
    mij = to_mij(rho,v,w,kappa,sigma,h) #in up-south-east convention
    ## noise amplitude
    amp = (m[6:(6+ns)]+MAXVAL)/720 + 0.0001 #[0,10]
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[6+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))

    ##data residual between observations and predictions
    res = data - pred_shifted
    
    lp1 = np.einsum('...i,...i', np.einsum('...i,...ij', res, cov_inv), res)
    lp1 /= (noise_amp**2 * np.exp(log_cov_det*2/nt))
    lp2 = 2 * log_cov_det + 2 * nt * np.log(noise_amp)
    ## return
    return -0.5 * np.sum(lp1 + lp2)
    
def log_prob_noisecov_timeshift_mij(m, data, greens, noise_std):#, cov_inv, log_cov_det):
    '''
    Full mt inversion using the mij parameterization and considering correlated data noise
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,ne,nt = greens.shape
    ##use the primitive MT in Bayesian sampling
    mij = m[:ne] #up-south-east convention

    ## noise amplitude
    amp = (m[ne:(ne+ns)]+MAXVAL)/720 + 0.0001 #0-10
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[ne+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    res = data - pred_shifted
    
    lp1 = np.einsum('...i,...i', np.einsum('...i,...ij', res, cov_inv), res)
    lp1 /= (noise_amp**2 * np.exp(log_cov_det*2/nt))
    lp2 = 2 * log_cov_det + 2 * nt * np.log(noise_amp)
    
    ## return
    return -0.5 * np.sum(lp1 + lp2)

def log_prob_noisecov_timeshift_TashiroMT(m, data, greens, noise_std):
    '''
    The case that uses the misfit function from mutq in which the time-shift is treated differently.
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,ne,nt = greens.shape
    ##use the primitive MT in Bayesian sampling
    m[:5] = (MAXVAL + m[:5]) / 7200 #x_i (1-5) ~ (0,1)
    m[5] = (m[5]+MAXVAL)/3600 + 4      #Mw 4-6
    mij = Tashiro2MT6(m[:6]) 
    mij = ned2rtp(mij) #up-south-east convention
    
    ## noise amplitude
    amp = (m[ne:(ne+ns)]+MAXVAL)/720 + 0.0001
    ## new_sigma
    noise_amp = (noise_std.T * amp).T
    
    #timeshift
    shift = m[6+ns:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    
    res =  data-pred_shifted
    lp1 = np.einsum('...i,...i', np.einsum('...i,...ij', res, cov_inv), res)
    lp1 /= (noise_amp**2 * np.exp(log_cov_det*2/nt))
    lp2 = 2 * log_cov_det + 2 * nt * np.log(noise_amp)
    
    ## return
    return -0.5 * np.sum(lp1 + lp2)

# def log_prob_noisecov_timeshift_mij(m, data, greens, noise_std):#, cov_inv, log_cov_det):
#     '''
#     Full mt inversion using the mij parameterization and considering correlated data noise
#     '''
#     if not np.isfinite(log_prior(m)): return -np.inf
    
#     ns,nc,ne,nt = greens.shape
#     ##use the primitive MT in Bayesian sampling
#     mij = m[:ne] #up-south-east convention

#     ## noise amplitude
#     amp = (m[ne:(ne+ns)]+MAXVAL)/720 + 0.0001 #0-10
#     ## new_sigma
#     noise_amp = (noise_std.T * amp).T
    
#     #timeshift
#     shift = m[ne+ns:] / 360 #[-10,10] s
    
#     ## d=G.m
#     pred = np.einsum('scet, e->sct', greens, mij)
#     pred_shifted = np.zeros(pred.shape)
#     omega = 2*np.pi*fftfreq(nt, d=1.0)
#     ##V/R share the same timeshift, T has another timeshift
#     for s in range(ns):
#         pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
#         pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
#     res = data - pred_shifted
    
#     lp1 = np.einsum('...i,...i', np.einsum('...i,...ij', res, cov_inv), res)
#     lp1 /= (noise_amp**2) * 2
#     lp2 = 3*nt*np.log(noise_amp)
    
#     ## return
#     return -1*np.sum(lp1 + lp2)

def log_prob_timeshift_mij(m, data, greens, noise_std):#, cov_inv, log_cov_det):
    '''
    Full mt inversion using the mij parameterization and considering correlated data noise
    '''
    if not np.isfinite(log_prior(m)): return -np.inf
    
    ns,nc,ne,nt = greens.shape
    ##use the primitive MT in Bayesian sampling
    mij = m[:ne] #up-south-east convention

    ## new_sigma
    noise_amp = noise_std
    
    #timeshift
    shift = m[ne:] / 360 #[-10,10] s
    
    ## d=G.m
    pred = np.einsum('scet, e->sct', greens, mij)
    pred_shifted = np.zeros(pred.shape)
    omega = 2*np.pi*fftfreq(nt, d=1.0)
    ##V/R share the same timeshift, T has another timeshift
    for s in range(ns):
        pred_shifted[s,:2] = np.real(ifft(fft(pred[s,:2], axis=1) * np.exp(-1j*omega*shift[2*s])))
        pred_shifted[s,2] = np.real(ifft(fft(pred[s,2]) * np.exp(-1j*omega*shift[2*s+1])))
    res = data - pred_shifted
    
    lp1 = np.einsum('...i,...i', np.einsum('...i,...ij', res, cov_inv), res)
    lp1 /= (noise_amp**2 * np.exp(log_cov_det*2/nt))
    lp2 = 2 * log_cov_det + 2 * nt * np.log(noise_amp)
    ## return
    return -0.5 * np.sum(lp1 + lp2)