import numpy as np
from copy import deepcopy
from mtuq.util.math import to_mij, to_rtp
from mtuq.util.signal import get_components, get_time_sampling
from mtuq.misfit.waveform import level2 as level2
from numpy.fft import fft, ifft, fftfreq

##this function copy most definition of level2.py in mtuq
def to_numpy_arrays(mtuq_data, mtuq_greens):
    """
    this function converts the structured data and greens from mtuq to NumPy arrays
    """
    #
    # collect metadata
    stations = level2._get_stations(mtuq_data)
    components = level2._get_components(mtuq_data)

    ns = len(stations)
    nc = len(components)
    
    # collapse main structures into NumPy arrays
    data = level2._get_data(mtuq_data, stations, components)       #ns x nc x nt
    greens = level2._get_greens(mtuq_greens, stations, components) #ns x nc x ne x nt in up-south-east convention
    #
    #mask the components during log_likelihood: 0-use, 1-no use
    weight_mask = np.zeros((ns, nc))
    for _i in range(ns):
        for _j in range(nc):
            # mask this component if all elements are zeros
            if not np.any(data[_i, _j]):
                weight_mask[_i, _j] = 1

    return data, greens, weight_mask

def shift_data(data, tau):
    ns = len(data)
    nt = data[0][0].stats.npts
    dt = data[0][0].stats.delta
    omega = 2*np.pi*fftfreq(nt, d=dt)
    shift = -1 * tau
    for s in range(ns):
        data[s][0].data = np.real(ifft(fft(data[s][0].data) * np.exp(-1j*omega*shift[2*s])))
        data[s][1].data = np.real(ifft(fft(data[s][1].data) * np.exp(-1j*omega*shift[2*s])))
        data[s][2].data = np.real(ifft(fft(data[s][2].data) * np.exp(-1j*omega*shift[2*s+1])))    
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
