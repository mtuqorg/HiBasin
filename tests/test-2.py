import sys
from mtuq.util.math import to_delta_gamma,to_mij, to_rho, to_v_w
sys.path.insert(0, '/Users/hujy/Documents/Research/BayMTI/src/')
from utils.math import MT2Tashiro,rtp2ned, to_lune, Tashiro2MT6
import numpy as np
from pyrocko.moment_tensor import MomentTensor


tashiro_mt = [0.2, 0.1, 0.4, 0.7, 0.1, 5.0]
mt = Tashiro2MT6(tashiro_mt)
new_mt = MT2Tashiro(mt)

print("raw mt:", tashiro_mt)
print("new mt:", new_mt)
