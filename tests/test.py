import sys
from mtuq.util.math import to_delta_gamma,to_mij, to_rho, to_v_w
sys.path.insert(0, '/Users/hujy/Documents/Research/BayMTI/src/')
from utils.math import rtp2ned, to_lune, Tashiro2MT6
import numpy as np
from pyrocko.moment_tensor import MomentTensor


tape_mt = [1.0e+17, -0.2, 0.4, 80, -70, 0.5]
mij = to_mij(1.0e+17, -0.2, 0.4, 80, -70, 0.5) #Up-South-East convention
#mij = rtp2ned(mij)
lune_mt = to_lune(mij)

mt= MomentTensor.from_values(mij)

print('Tape:', tape_mt)
print('lune:', to_delta_gamma(-0.2,0.4))
print("mij:", mij)
print('Tape_recovered:',lune_mt)
print('new mij:', to_mij(lune_mt[0],lune_mt[1],lune_mt[2],lune_mt[3],lune_mt[4],lune_mt[5]))
