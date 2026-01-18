from matplotlib import rcParams

rcParams["savefig.dpi"] = 300
rcParams["figure.dpi"] = 200
rcParams["font.size"] = 15

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import numpy as np
import os
import glob
import corner
import emcee
from netCDF4 import Dataset
import emcee
from numpy.fft import rfft, irfft, rfftfreq
import os
import multiprocessing
import matplotlib
from obspy.imaging.mopad_wrapper import beach

import os
import numpy as np

from mtuq import read, open_db, download_greens_tensors
from mtuq.event import Origin
from mtuq.graphics import plot_data_greens1, plot_beachball, plot_misfit_lune
from mtuq.grid import FullMomentTensorGridSemiregular
from mtuq.grid_search import grid_search
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.grid import UnstructuredGrid
from mtuq.grid.moment_tensor import to_mt
from mtuq.util.math import to_mij, to_rho
from mtuq.util.cap import taper

import multiprocessing
import emcee
import sys
sys.path.insert(0, '/Users/hujy/Documents/Research/BayMTI/src/')
from likelihood import *
from utils.data_selection import data_noise_estimate_uncorrelated, get_solution
from utils.misfit_preparation import shift_greens, misfit_preparation
from utils.visualization import plot_waveform_fit
from obspy.signal.filter import bandpass

def ned2rtp(mt_ned):
    Mxx,Myy,Mzz, Mxy,Mxz,Myz = mt_ned
    Mrr=Mzz
    Mtt=Mxx
    Mpp=Myy
    Mrt=Mxz
    Mrp=-Myz
    Mtp=-Mxy
    return np.array([Mrr, Mtt, Mpp, Mrt, Mrp, Mtp])


if __name__=='__main__':

    path_data=    fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/*.[zrt]')
    path_weights= fullpath('/Users/hujy/Documents/Research/BayMTI/data/20090407201255351/weights.dat')
    event_id=     '20090407201255351'
    model=        'ak135'

    station_id_list = parse_station_codes(path_weights)

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454,
        'longitude': -149.743,
        'depth_in_m': 5000.,
        })


    print('Reading data...\n')
    data = read(path_data, format='sac', 
        event_id=event_id,
        station_id_list=station_id_list,
        tags=['units:m', 'type:velocity']) 


    data.sort_by_distance()
    stations = data.get_stations()


    ## MAP BACKGROUND
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    m = Basemap(projection='merc',llcrnrlat=58,urcrnrlat=64,\
                llcrnrlon=-155,urcrnrlon=-140,lat_ts=45,resolution='i', ax=ax)
    #m.etopo()
    #m.shadedrelief()
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray')
    m.drawparallels(np.arange(58, 64.1,2.), dashes=[3, 2], linewidth=.5, labels=[1,0,0,0], fontsize=12)
    m.drawmeridians(np.arange(-155, -140.1, 5.), dashes=[3, 2], linewidth=.5, labels=[0,0,0,1], fontsize=12)
    # ## LOCATION OF NUCLEAR TESTS
    xx_0, yy_0 = m(origin.longitude, origin.latitude)
    ## LOCATION OF STATIONS
    xx, yy = m([s.longitude for s in stations], [s.latitude for s in stations])

    for x, y, s in zip(xx, yy, [s.station for s in stations]):
        plt.plot([xx_0,x],[yy_0,y], color = 'white')
        plt.text(x, y+10e3, s, fontsize=10, bbox={'boxstyle':'round', 'pad':0.05, 'color':'w', 'ec':'none', 'alpha':0.75})
    plt.plot(xx, yy, 'v', color='b', markersize=7)    
    plt.plot(xx_0, yy_0, '*', color='r', markersize=10)
    plt.tight_layout()
    plt.savefig("station_source_map.jpg", bbox_inches='tight',dpi=300)
   # plt.show()
