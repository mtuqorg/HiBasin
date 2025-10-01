import numpy as np
import time
from copy import deepcopy
from mtuq.util.math import to_mij, to_rtp
from mtuq.util.signal import get_components, get_time_sampling
from mtuq.misfit.waveform import level2 as level2
from numpy.fft import fft, ifft, fftfreq

##this function copy most definition of level2.py in mtuq
def misfit_preparation(data, greens):
    """
    Data misfit function (fast Python/C version)

    See ``mtuq/misfit/waveform/__init__.py`` for more information
    """

    #
    # collect metadata
    #
    nt, dt = level2._get_time_sampling(data)
    stations = level2._get_stations(data)
    components = level2._get_components(data)

    #
    # collapse main structures into NumPy arrays
    #
    data = level2._get_data(data, stations, components)       #ns x nc x nt
    greens = level2._get_greens(greens, stations, components) #ns x nc x ne x nt in up-south-east convention

    #
    return data, greens


def shift_data(data, tau):
    ns = len(data)
    nt = data[0][0].stats.npts
    dt = data[0][0].stats.delta
    omega = 2*np.pi*fftfreq(nt, d=dt)
    shift = tau
    for s in range(ns): 
        data_z = data[s].select(component='Z')[0].data 
        data_r = data[s].select(component='R')[0].data
        data_t = data[s].select(component='T')[0].data

        data[s].select(component='Z')[0].data = np.real(ifft(fft(data_z) * np.exp(-1j*omega*shift[2*s])))
        data[s].select(component='R')[0].data = np.real(ifft(fft(data_r) * np.exp(-1j*omega*shift[2*s])))
        data[s].select(component='T')[0].data = np.real(ifft(fft(data_t) * np.exp(-1j*omega*shift[2*s+1])))
    return data

def shift_greens(greens, tau):
    ns = len(greens)
    ne = len(greens[0])
    nt = greens[0][0].stats.npts
    dt = greens[0][0].stats.delta
    omega = 2*np.pi*fftfreq(nt, d=dt)
    shift = tau
    
    if ne == 9:
        for s in range(ns):
            for e in range(3):
                #Z component
                greens[s][3*e].data = np.real(ifft(fft(greens[s][3*e].data) * np.exp(-1j*omega*shift[2*s]))) 
                #R component
                greens[s][3*e+1].data = np.real(ifft(fft(greens[s][3*e+1].data) * np.exp(-1j*omega*shift[2*s])))
                #T component
                greens[s][3*e+2].data = np.real(ifft(fft(greens[s][3*e+2].data) * np.exp(-1j*omega*shift[2*s+1])))
    else:
        for s in range(ns):
            for e in range(4):
                #Z component
                greens[s][e].data = np.real(ifft(fft(greens[s][e].data) * np.exp(-1j*omega*shift[2*s]))) 
                #R component
                greens[s][e+4].data = np.real(ifft(fft(greens[s][e+4].data) * np.exp(-1j*omega*shift[2*s])))
              #T component
            for e in range(2):
                  greens[s][e+8].data = np.real(ifft(fft(greens[s][e+8].data) * np.exp(-1j*omega*shift[2*s+1])))
    return greens
