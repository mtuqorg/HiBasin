import sys
import numpy as np

from mtuq.util.math import to_mij, to_rho, to_Mw,to_rtp, to_delta_gamma
sys.path.insert(0, '/Users/hujy/Documents/Research/BayMTI/src/')
from utils.math import to_lune

def ned2rtp(mt_ned):
    Mxx,Myy,Mzz, Mxy,Mxz,Myz = mt_ned
    Mrr=Mzz
    Mtt=Mxx
    Mpp=Myy
    Mrt=Mxz
    Mrp=-Myz
    Mtp=-Mxy
    return np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])

mt = ned2rtp(np.loadtxt('mt_input.txt'))
rho,v,w,kappa,sigma,h = to_lune(mt)

delta, gamma = to_delta_gamma(v,w)
print('delta: ', delta, 'gamma: ', gamma)
print('Mw: ', to_Mw(rho))

print("Best solution")
mt={'rho': 9.858373235167158e+16, 'v': -0.03886340096371379, 'w': 1.1697033883804389, 'kappa': 168.48025520971504, 'sigma': 89.29265893900006, 'h': 0.32036314256894}

delta, gamma = to_delta_gamma(mt['v'],mt['w'])
print('delta: ', delta, 'gamma: ', gamma)

