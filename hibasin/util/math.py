import numpy as np
import os
import matplotlib.pyplot as plt
import time
from numpy.random import uniform, normal, standard_normal, standard_cauchy, randint
from copy import deepcopy
from obspy.signal.filter import bandpass
from netCDF4 import Dataset
import pyrocko.moment_tensor as mtm
from pyrocko.moment_tensor import MomentTensor
from scipy.linalg import cholesky, solve_triangular
from scipy.signal import correlate
from mtuq.util.math import to_mij, to_rho, to_v_w,to_M0
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

MAXVAL=3600

def map_1D_array_to_ranges(arr, new_min, new_max, old_min=-3600, old_max=3600):
    """
    Map a 1D array to a specified range [low, high].
    """
    return new_min + (new_max - new_min) * (arr - old_min) / (old_max - old_min)
##
def rtp2ned(mt_rtp):
    Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = mt_rtp
    Mzz=Mrr
    Mxx=Mtt
    Myy=Mpp
    Mxz=Mrt
    Myz=-Mrp
    Mxy=-Mtp
    return np.array([Mxx,Myy,Mzz, Mxy,Mxz,Myz])

def rtp2ned2(Mrr, Mtt, Mpp, Mrt, Mrp, Mtp):
    Mzz=Mrr
    Mxx=Mtt
    Myy=Mpp
    Mxz=Mrt
    Myz=-Mrp
    Mxy=-Mtp
    
    if type(Mrr) is np.ndarray:
        return np.column_stack([Mxx,Myy,Mzz, Mxy,Mxz,Myz])
    else:
        return np.array([Mxx,Myy,Mzz, Mxy,Mxz,Myz])    

def ned2rtp(mt_ned):
    Mxx,Myy,Mzz, Mxy,Mxz,Myz = mt_ned
    Mrr=Mzz
    Mtt=Mxx
    Mpp=Myy
    Mrt=Mxz
    Mrp=-Myz
    Mtp=-Mxy
    return np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])

def ned2rtp2(Mxx,Myy,Mzz,Mxy,Mxz,Myz):
    Mrr=Mzz
    Mtt=Mxx
    Mpp=Myy
    Mrt=Mxz
    Mrp=-Myz
    Mtp=-Mxy
    
    if type(Mrr) is np.ndarray:
        return np.column_stack([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])
    else:
        return np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])

def MT6toMT9(mt):
    mt9 = np.zeros((3,3))
    mt9[0,:] = np.array([mt[0], mt[3], mt[4]])
    mt9[1,:] = np.array([mt[3], mt[1], mt[5]])
    mt9[2,:] = np.array([mt[4], mt[5], mt[2]])
    return mt9

##The following functions were developped by Son Pham (son.pham@anu.edu.au)
def MT9toNatural(mt9):
    pyrocko_mt = MomentTensor(m=mt9)
    eigenvals = pyrocko_mt.eigenvals()
    eigenvals.sort()
    lambda1, lambda2, lambda3 = eigenvals[::-1] # lamda1 >= lambda2 >= lambda3
    ## Coordinates of source-type lune diagram
    rho = np.sqrt(np.sum(eigenvals.dot(eigenvals)))
    gamma = np.arctan2((-lambda1+2*lambda2-lambda3), (np.sqrt(3)*(lambda1-lambda3))) # longitude
    beta= np.arccos((lambda1+lambda2+lambda3) / (np.sqrt(3)*rho))
    sigma = np.pi/2 - beta # latitude
    strike, dip, rake = pyrocko_mt.both_strike_dip_rake()[0]
    return (np.degrees(gamma), np.degrees(sigma), strike, dip, rake, pyrocko_mt.moment_magnitude())

def mt2lune(mxx, myy, mzz, mxy, mxz, myz):
    m33 = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]])
    eivals = np.linalg.eigvals(m33)
    eivals.sort()
    
    ## lune longitude calculated from the eigen value triple
    nom = -eivals[0] - eivals[2] + 2 * eivals[1]
    den = np.sqrt(3) * (eivals[2]- eivals[0])
    gamma = np.arctan2(nom, den) / np.pi * 180

    ## lune latitude calculated from the eigen value triple
    nom = np.sum(eivals)
    den = np.sqrt(3) * np.sqrt(np.sum(eivals**2))
    beta = np.arccos(nom / den) / np.pi * 180

    ## orientation angles determined from the eigen vector triple
    return gamma, 90 - beta

def to_lune(mij):
    """
    Convert moment tensor parameters (in up-south-east convention) to 2015Tape parameters
    """
    mt1,mt2,mt3,mt4,mt5,mt6 = rtp2ned(mij)
    pyrocko_mt = MomentTensor.from_values((mt1,mt2,mt3,mt4,mt5,mt6))
    eigenvals = pyrocko_mt.eigenvals()
    eigenvals.sort()
    lambda1, lambda2, lambda3 = eigenvals[::-1] # lamda1 >= lambda2 >= lambda3
    ## Coordinates of source-type lune diagram
    rho = np.sqrt(np.sum(eigenvals.dot(eigenvals)))
    gamma = np.arctan2((-lambda1+2*lambda2-lambda3), (np.sqrt(3)*(lambda1-lambda3))) # longitude
    beta= np.arccos((lambda1+lambda2+lambda3) / (np.sqrt(3)*rho))
    delta = np.pi/2 - beta # latitude
    strike, dip, rake = pyrocko_mt.both_strike_dip_rake()[0]
    
    M0 = pyrocko_mt.scalar_moment()
    Mw = 2/3*np.log10(M0) - 18.2/3 #the formular used in MTUQ, a little different from pyrocko
    
    v, w = to_v_w(np.rad2deg(delta), np.rad2deg(gamma))
    rho = to_rho(Mw)
    kappa = strike
    sigma = rake
    h = np.cos(np.radians(dip))
 
    return rho, v, w, kappa, sigma, h

def ConvertChain_Tashiro2MT6(chain_fname):
    fm_chain = np.loadtxt(chain_fname, usecols=range(6))
    num_steps = fm_chain.shape[0]

    fid = open(chain_fname.replace('.txt', '.mt6'), 'w')
    ## Converting a focal mechanism from Tashiro's uniform coordinates to 
    # the natural cooridnates with strike, dip, rake angles and lune coordinates
    for i_step in range(0, num_steps):
        pyrocko_mt = MomentTensor(m=Tashiro2MT9(fm_chain[i_step,:]))
        eigenvals = pyrocko_mt.eigenvals()
        eigenvals.sort()
        lambda1, lambda2, lambda3 = eigenvals[::-1] # lamda1 >= lambda2 >= lambda3
        ## Coordinates of source-type lune diagram
        rho = np.sqrt(np.sum(eigenvals.dot(eigenvals)))
        gamma = np.arctan2((-lambda1+2*lambda2-lambda3), (np.sqrt(3)*(lambda1-lambda3)))
        beta = np.arccos((lambda1+lambda2+lambda3) / (np.sqrt(3)*rho))
        sigma = np.pi/2 - beta
        ## MT6 coleection for beachball plots
        tmp = np.append(pyrocko_mt.m6(), [np.degrees(gamma), np.degrees(sigma), pyrocko_mt.magnitude])
        fmt = '%e '*len(tmp) + '\n'
        fid.write(fmt % tuple(tmp))
    fid.close()

def Tashiro2Natural(fm):
    pyrocko_mt = MomentTensor(m=Tashiro2MT9(fm))
    eigenvals = pyrocko_mt.eigenvals()
    eigenvals.sort()
    lambda1, lambda2, lambda3 = eigenvals[::-1] # lamda1 >= lambda2 >= lambda3
    ## Coordinates of source-type lune diagram
    rho = np.sqrt(np.sum(eigenvals.dot(eigenvals)))
    gamma = np.arctan2((-lambda1+2*lambda2-lambda3), (np.sqrt(3)*(lambda1-lambda3))) # longitude
    beta= np.arccos((lambda1+lambda2+lambda3) / (np.sqrt(3)*rho))
    sigma = np.pi/2 - beta # latitude
    strike, dip, rake = pyrocko_mt.both_strike_dip_rake()[0]
    return (np.degrees(gamma), np.degrees(sigma), strike, dip, rake, pyrocko_mt.moment_magnitude())

def Tashiro2MT6(fm):
    '''
    Uniform parameterization of moment tensor using Tashiro (1977) method.
    This parameterization was also used in Stahler & Sigloch (2014).
    '''
#     assert (fm.shape[0] == 6)
    xx1, xx2, xx3, xx4, xx5, Mw = fm
#     M0 = mtm.magnitude_to_moment(Mw)
    M0 = 10.**(1.5*Mw + 9.1) #used by MTUQ
    Y3 = 1
    Y2 = np.sqrt(xx2) * Y3
    Y1 = Y2 * xx1
    Mxx = M0 * np.sqrt(Y1) * np.cos(2*np.pi*xx3) * np.sqrt(2)
    Myy = M0 * np.sqrt(Y1) * np.sin(2*np.pi*xx3) * np.sqrt(2)
    Mzz = M0 * np.sqrt(Y2-Y1) * np.cos(2*np.pi*xx4) * np.sqrt(2)
    Mxy = M0 * np.sqrt(Y2-Y1) * np.sin(2*np.pi*xx4)
    Mxz = M0 * np.sqrt(Y3-Y2) * np.cos(2*np.pi*xx5)
    Myz = M0 * np.sqrt(Y3-Y2) * np.sin(2*np.pi*xx5)
    return np.array([Mxx, Myy, Mzz, Mxy, Mxz, Myz])

def Tashiro2MT6_vec(xx1, xx2, xx3, xx4, xx5, Mw):
    '''
    Uniform parameterization of moment tensor using Tashiro (1977) method.
    This parameterization was also used in Stahler & Sigloch (2014).
    '''
#     M0 = mtm.magnitude_to_moment(Mw)
    M0 = 10.**(1.5*Mw + 9.1) #used by MTUQ
    Y3 = 1
    Y2 = np.sqrt(xx2) * Y3
    Y1 = Y2 * xx1
    Mxx = M0 * np.sqrt(Y1) * np.cos(2*np.pi*xx3) * np.sqrt(2)
    Myy = M0 * np.sqrt(Y1) * np.sin(2*np.pi*xx3) * np.sqrt(2)
    Mzz = M0 * np.sqrt(Y2-Y1) * np.cos(2*np.pi*xx4) * np.sqrt(2)
    Mxy = M0 * np.sqrt(Y2-Y1) * np.sin(2*np.pi*xx4)
    Mxz = M0 * np.sqrt(Y3-Y2) * np.cos(2*np.pi*xx5)
    Myz = M0 * np.sqrt(Y3-Y2) * np.sin(2*np.pi*xx5)
    
    if type(xx1) is np.ndarray:
        return np.column_stack([Mxx, Myy, Mzz, Mxy, Mxz, Myz])
    else:
        return np.array([Mxx, Myy, Mzz, Mxy, Mxz, Myz])

def Tashiro2MT9(fm):
    '''
    Uniform parameterization of moment tensor using Tashiro (1977) method.
    This parameterization was also used in Stahler & Sigloch (2014).
    '''
    assert (fm.shape[0] == 6)
    xx1, xx2, xx3, xx4, xx5, Mw = fm
    M0 = mtm.magnitude_to_moment(Mw)
    Y3 = 1
    Y2 = np.sqrt(xx2)
    Y1 = Y2 * xx1
    Mxx = M0 * np.sqrt(Y1) * np.cos(2*np.pi*xx3) * np.sqrt(2)
    Myy = M0 * np.sqrt(Y1) * np.sin(2*np.pi*xx3) * np.sqrt(2)
    Mzz = M0 * np.sqrt(Y2-Y1) * np.cos(2*np.pi*xx4) * np.sqrt(2)
    Mxy = M0 * np.sqrt(Y2-Y1) * np.sin(2*np.pi*xx4)
    Mxz = M0 * np.sqrt(Y3-Y2) * np.cos(2*np.pi*xx5)
    Myz = M0 * np.sqrt(Y3-Y2) * np.sin(2*np.pi*xx5)
    return np.array([[Mxx, Mxy, Mxz], [Mxy, Myy, Myz], [Mxz, Myz, Mzz]])

def MT2Tashiro(mt6):
    ## Normalize MT
    mxx, myy, mzz, mxy, mxz, myz = mt6
    M0 = np.sqrt((mxx**2 + myy**2 + mzz**2)/2 + mxy**2 + mxz**2 + myz**2)
    mxx, myy, mzz, mxy, mxz, myz = np.array(mt6) / M0

    ##
    Y1 = (mxx**2 + myy**2)/2
    Y2 = Y1 + (mzz**2/2 + mxy**2)
    Y3 = 1

    x1 = Y1 / Y2
    x2 = (Y2 / Y3)**2
    x3 = np.arctan2(myy, mxx)
    if x3 < 0: x3 += (2*np.pi)
    x4 = np.arctan2(mxy, mzz/np.sqrt(2))
    if x4 < 0: x4 += (2*np.pi)
    x5 = np.arctan2(myz, mxz)
    if x5 < 0: x5 += (2*np.pi)

    return np.array([x1, x2, x3/(2*np.pi), x4/(2*np.pi), x5/(2*np.pi), mtm.moment_to_magnitude(M0)])

##copy from mtuq
def to_mij_rev(m):
    """ Converts from lune parameters to moment tensor parameters
    (up-south-east convention)
    """
    v = m[0] / 10800
    w = m[1] * np.pi / 9600             
    kappa = (m[2] + MAXVAL) / 20             ##0, 360
    sigma = m[3] / 40                        ##-90, 90
    h = (m[4] + MAXVAL) / 7200               ##cos(dip)
    rho = to_rho((m[5]+MAXVAL)/3600 + 4)     ##Mw: 4-6

    kR3 = np.sqrt(3.)
    k2R6 = 2.*np.sqrt(6.)
    k2R3 = 2.*np.sqrt(3.)
    k4R6 = 4.*np.sqrt(6.)
    k8R6 = 8.*np.sqrt(6.)

    m0 = rho/np.sqrt(2.)

    # delta, gamma = to_delta_gamma(v, w)
    #calculate delta
    beta0 = np.linspace(0, np.pi, 100)
    u0 = 0.75*beta0 - 0.5*np.sin(2.*beta0) + 0.0625*np.sin(4.*beta0)
    beta = np.interp(3.*np.pi/8. - w, u0, beta0)
    delta = np.rad2deg(np.pi/2. - beta)
    beta = 90. - delta

    #calculate gamma
    gamma = np.rad2deg((1./3.)*np.arcsin(3.*v))
    
    gamma = np.deg2rad(gamma)
    beta = np.deg2rad(beta)
    kappa = np.deg2rad(kappa)
    sigma = np.deg2rad(sigma)
    theta = np.arccos(h)

    Cb  = np.cos(beta)
    Cg  = np.cos(gamma)
    Cs  = np.cos(sigma)
    Ct  = np.cos(theta)
    Ck  = np.cos(kappa)
    C2k = np.cos(2.0*kappa)
    C2s = np.cos(2.0*sigma)
    C2t = np.cos(2.0*theta)

    Sb  = np.sin(beta)
    Sg  = np.sin(gamma)
    Ss  = np.sin(sigma)
    St  = np.sin(theta)
    Sk  = np.sin(kappa)
    S2k = np.sin(2.0*kappa)
    S2s = np.sin(2.0*sigma)
    S2t = np.sin(2.0*theta)

    mt0 = m0 * (1./12.) * \
        (k4R6*Cb + Sb*(kR3*Sg*(-1. - 3.*C2t + 6.*C2s*St*St) + 12.*Cg*S2t*Ss))

    mt1 = m0* (1./24.) * \
        (k8R6*Cb + Sb*(-24.*Cg*(Cs*St*S2k + S2t*Sk*Sk*Ss) + kR3*Sg * \
        ((1. + 3.*C2k)*(1. - 3.*C2s) + 12.*C2t*Cs*Cs*Sk*Sk - 12.*Ct*S2k*S2s)))

    mt2 = m0* (1./6.) * \
        (k2R6*Cb + Sb*(kR3*Ct*Ct*Ck*Ck*(1. + 3.*C2s)*Sg - k2R3*Ck*Ck*Sg*St*St +
        kR3*(1. - 3.*C2s)*Sg*Sk*Sk + 6.*Cg*Cs*St*S2k +
        3.*Ct*(-4.*Cg*Ck*Ck*St*Ss + kR3*Sg*S2k*S2s)))

    mt3 = m0* (-1./2.)*Sb*(k2R3*Cs*Sg*St*(Ct*Cs*Sk - Ck*Ss) +
        2.*Cg*(Ct*Ck*Cs + C2t*Sk*Ss))

    mt4 = -m0* (1./2.)*Sb*(Ck*(kR3*Cs*Cs*Sg*S2t + 2.*Cg*C2t*Ss) +
        Sk*(-2.*Cg*Ct*Cs + kR3*Sg*St*S2s))

    mt5 = -m0* (1./8.)*Sb*(4.*Cg*(2.*C2k*Cs*St + S2t*S2k*Ss) +
        kR3*Sg*((1. - 2.*C2t*Cs*Cs - 3.*C2s)*S2k + 4.*Ct*C2k*S2s))

    return np.array([mt0, mt1, mt2, mt3, mt4, mt5])
        
def numerical_jacobian(f, x, eps=1e-6):
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        J[:, i] = (f(x_eps) - f(x)) / eps
    return J

def exponential_covariance(length, scale=10):
    '''
    Generate the covariance matrix for exponential decay noise model
    '''
    x = np.arange(length)
    cov_matrix = np.exp(-np.abs(x[:, None] - x[None, :]) / scale)
    return cov_matrix
    
def calc_InversionDeterminant_cd(cov_d):
    '''
    Compute the inverse of matrix N-by-N cov_d, where N is the number of samples
    '''
    ns,nc,nt,_ = cov_d.shape
    cov_inv = np.zeros((ns,nc,nt,nt))
    log_cov_det = np.zeros((ns,nc))
    # Cholesky decomposition to obtain lower matrix
    for ist in range(ns):
        for ic in range(nc):
            cov = cov_d[ist,ic] 
            covL = cholesky(cov, lower=True)
            #log of sqrt determinant
            factor = np.sum(np.log(np.abs(np.diag(covL))))
            covL /= np.exp(factor / nt)
            
            # Invert combined matrix
            covL_inv = solve_triangular(covL, np.eye(nt), lower=True)
            cov_inv[ist,ic] = np.matmul(covL_inv.T, covL_inv)
            log_cov_det[ist,ic] = factor
    return cov_inv, log_cov_det

def ensemble_uncertainty(samples, labels=None):
    """
    Compute and print mean ± std for each parameter in a posterior ensemble.

    Parameters
    ----------
    samples : array-like, shape (n_samples,) or (n_samples, n_params)
        Posterior samples. Can be MT components, noise amplitudes, or time shifts.
    labels : list of str, optional
        Parameter names, one per column. Defaults to 'param_0', 'param_1', ...

    Returns
    -------
    means : ndarray, shape (n_params,)
    stds  : ndarray, shape (n_params,)
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if samples.shape[0] == 1:
        samples = samples.T        # handle 1-D input passed as row vector
    n_params = samples.shape[1]

    if labels is None:
        labels = ['param_%d' % i for i in range(n_params)]

    means = np.mean(samples, axis=0)
    stds  = np.std(samples,  axis=0)

    max_label_len = max(len(l) for l in labels)
    for lbl, mu, sigma in zip(labels, means, stds):
        print('  %-*s  %g ± %g' % (max_label_len, lbl, mu, sigma))

    return means, stds


def ensemble_mt_decomposition(samples_ned, thin=10):
    """
    Decompose each MT sample in a posterior ensemble into ISO, CLVD, and DC
    components using pyrocko's standard decomposition, then report
    mean ± std of each fraction and moment.

    Parameters
    ----------
    samples_ned : array-like, shape (n_samples, 6)
        MT samples in NED convention: [Mxx, Myy, Mzz, Mxy, Mxz, Myz],
        as returned by rtp2ned / rtp2ned2.
    thin : int
        Use every `thin`-th sample to keep computation manageable.

    Returns
    -------
    fracs : ndarray, shape (n_used, 3)
        Columns: [ISO %, DC %, CLVD %] for each (thinned) sample.
    """
    samples = np.asarray(samples_ned, dtype=float)[::thin]
    n = samples.shape[0]

    fracs   = np.zeros((n, 3))   # ISO %, DC %, CLVD %
    moments = np.zeros((n, 3))   # ISO, DC, CLVD scalar moments (N-m)

    for i in range(n):
        mt = MomentTensor.from_values(samples[i])
        d  = mt.standard_decomposition()
        fracs[i]   = [100 * d[0][1], 100 * d[1][1], 100 * d[2][1]]
        moments[i] = [d[0][0],       d[1][0],        d[2][0]]

    print('--- MT decomposition: fractions (mean +/- std) [%] ---')
    ensemble_uncertainty(fracs,   labels=['ISO', 'DC', 'CLVD'])
    print('--- MT decomposition: scalar moments (mean +/- std) [N-m] ---')
    ensemble_uncertainty(moments, labels=['ISO', 'DC', 'CLVD'])

    return fracs


def cc_optimal_shifts(solver, ref_mij):
    """
    Compute cross-correlation optimal time shift per station and component
    group using a reference moment tensor.

    Returns tau_cc of shape (time_shift_groups * ns,) in physical seconds,
    interleaved as [ZR_0, T_0, ZR_1, T_1, ...] for time_shift_groups=2,
    clipped to [time_shift_min, time_shift_max].
    """
    pred   = np.einsum('scet,e->sct', solver.greens, ref_mij)
    n_grps = solver.time_shift_groups
    tau    = np.zeros(n_grps * solver.ns)

    for s in range(solver.ns):
        cc_zr    = np.zeros(2 * solver.nt - 1)
        n_active = 0
        for c in range(min(2, solver.nc)):
            if not solver.weight_mask[s, c]:
                cc_zr    += correlate(solver.obs[s, c], pred[s, c], mode='full')
                n_active += 1
        shift_zr = float(
            (cc_zr.argmax() - (solver.nt - 1)) * solver.delta
            if n_active > 0 else 0.0)
        shift_zr = float(np.clip(shift_zr, solver.time_shift_min, solver.time_shift_max))

        if n_grps == 1:
            tau[s] = shift_zr
        else:
            tau[2 * s] = shift_zr
            shift_t = 0.0
            if solver.nc > 2 and not solver.weight_mask[s, 2]:
                cc_t    = correlate(solver.obs[s, 2], pred[s, 2], mode='full')
                shift_t = float((cc_t.argmax() - (solver.nt - 1)) * solver.delta)
                shift_t = float(np.clip(shift_t, solver.time_shift_min, solver.time_shift_max))
            tau[2 * s + 1] = shift_t

    return tau


def calcInversionDeterminant(cov_d):
    #- Diagonal data covariance matrices for each component
    '''
    Compute the inverse of matrix N-by-N cov_d, where N is the number of samples
    '''
    ns,nc,nt,_ = cov_d.shape
    cov_inv = np.zeros((ns,nc,nt,nt))
    cov_det_sqrt = np.zeros((ns,nc))
    
    for ist in range(ns):
        for ic in range(nc):
            cov = cov_d[ist, ic]
            # Cholesky decomposition to obtain lower matrix
            CovL = cholesky(cov, lower=True)
            factor = np.prod(np.diag(CovL))
            CovL /= factor**(1/nt)
            # Invert combined matrix
            CovL_inv = solve_triangular(CovL, np.eye(nt), lower=True)

            cov_inv[ist,ic] = np.matmul(CovL_inv.T, CovL_inv)
            cov_det_sqrt[ist,ic] = factor
    return cov_inv, cov_det_sqrt