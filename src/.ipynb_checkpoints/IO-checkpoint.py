from distutils.command.config import config
import numpy as np
from mtuq import Station, Origin, MomentTensor
from obspy import UTCDateTime, Stream
from instaseis import Source, Receiver
import instaseis
from mtuq.util.signal import cut, get_arrival, m_to_deg
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt

def read_stations(config_fn):
    stats = []
    with open(config_fn) as fid:
        for line in fid:
            tokens = line[:line.find('#')].split()
            if len(tokens) == 0: continue
            network, station = tokens[0].split('.')
            latitude = float(tokens[1])
            longitude = float(tokens[2])
            s = Station(network=network, station=station, latitude=latitude, longitude=longitude)
            stats.append(s)
    return stats

def read_event(evla, evlo, evdp, cmtsol_fn=None):
    # evla, evlo = 41.3199, 129.0491
    # evdp = 10 # km

    return Origin(latitude=evla, longitude=evlo, depth_in_m=evdp*1e3)

def download_greens_tensor(    
        instaseis_db,
        epicentral_distance_in_degree,
        source_depth_in_m,
        origin_time=UTCDateTime(0),
        kind="displacement",
        return_obspy_stream=True,
        dt=None,
        kernelwidth=12,
        definition="seiscomp"):
     
       src_latitude, src_longitude = 90.0, 0.0
       rec_latitude, rec_longitude = 90.0 - epicentral_distance_in_degree, 0.0

       # sources according to https://github.com/krischer/instaseis/issues/8
       # transformed to r, theta, phi
       #
       # Mtt =  Mxx, Mpp = Myy, Mrr =  Mzz
       # Mrp = -Myz, Mrt = Mxz, Mtp = -Mxy
       #
       # Mrr   Mtt   Mpp    Mrt    Mrp    Mtp
       #  0     0     0      0      0     -1.0    m1
       #  0     1.0  -1.0    0      0      0      m2
       #  0     0     0      0     -1.0    0      m3
       #  0     0     0      1.0    0      0      m4
       #  1.0   1.0   1.0    0      0      0      m6
       #  2.0  -1.0  -1.0    0      0      0      cl

       m1 = Source(
           src_latitude,
           src_longitude,
           source_depth_in_m,
           m_tp=-1.0,
           origin_time=origin_time,
       )
       m2 = Source(
           src_latitude,
           src_longitude,
           source_depth_in_m,
           m_tt=1.0,
           m_pp=-1.0,
           origin_time=origin_time,
       )
       m3 = Source(
           src_latitude,
           src_longitude,
           source_depth_in_m,
           m_rp=-1.0,
           origin_time=origin_time,
       )
       m4 = Source(
           src_latitude,
           src_longitude,
           source_depth_in_m,
           m_rt=1.0,
           origin_time=origin_time,
       )
       m6 = Source(
           src_latitude,
           src_longitude,
           source_depth_in_m,
           m_rr=1.0,
           m_tt=1.0,
           m_pp=1.0,
           origin_time=origin_time,
       )
       cl = Source(
           src_latitude,
           src_longitude,
           source_depth_in_m,
           m_rr=2.0,
           m_tt=-1.0,
           m_pp=-1.0,
           origin_time=origin_time,
       )

       receiver = Receiver(rec_latitude, rec_longitude)

       # Extract all seismograms - leverage the logic of the
       # get_seismograms() method as much as possible.
       args = {
           "receiver": receiver,
           "dt": dt,
           "kind": kind,
           "kernelwidth": kernelwidth,
           "return_obspy_stream": return_obspy_stream,
       }

       #the same order as in mtuq green_tensor       
       items = [
           ("ZSS", m2, "Z"),
           ("ZDS", m4, "Z"),
           ("ZDD", cl, "Z"),
           ("ZEP", m6, "Z"),
           ("RSS", m2, "R"),
           ("RDS", m4, "R"),
           ("RDD", cl, "R"),
           ("REP", m6, "R"),
           ("TSS", m1, "T"),
           ("TDS", m3, "T"),    
       ]

       if return_obspy_stream:
           st = Stream()
       else:
           st = {}

       for name, src, comp in items:
           if comp == 'Z':
               tr = instaseis_db.get_seismograms(source=src, components=comp, **args)
               tr.trim(origin_time, origin_time+1000)
               tr = tr[0]
               tr.stats.channel = name
               st.append(tr)
           else:
               tr = instaseis_db.get_seismograms(source=src, components='Z', **args)
               tr.trim(origin_time, origin_time+1000)
               #tr[0].data[:] = 0 #set all RT components to zeros. not required because only Z comp will be retrived
               tr = tr[0]
               tr.stats.channel = name
               st.append(tr)
           
       return st
    
def get_ak135f_1s_gfs(greens, config_fname_tel, source_depth_in_m, dt):
    db = instaseis.open_db("syngine://ak135f_1s")
    dist = np.loadtxt(config_fname_tel, usecols=[3], dtype=float)
    
    for i,idist in enumerate(dist):
        dist_in_deg = m_to_deg(idist*1000)
        gf=download_greens_tensor(db, epicentral_distance_in_degree=dist_in_deg, source_depth_in_m=source_depth_in_m, dt=dt)
        greens[i].traces = gf

    return greens

    
def processGreens(greens, config_fname_tel, window_bw, filter_band_tel, filter_zerophase_tel, filter_corners_tel):
   dist = np.loadtxt(config_fname_tel, usecols=[3], dtype=float)
   az = np.loadtxt(config_fname_tel, usecols=[4], dtype=float)
   model = TauPyModel(model='ak135')
   num_stats = len(dist)
   tele_greens_tensor =[]
   for i in range(num_stats):
       greens[i].detrend('demean') 
       greens[i].detrend('linear') 
       greens[i].taper(max_percentage=0.05)
       greens[i].filter('bandpass', freqmin=filter_band_tel[0], freqmax=filter_band_tel[1], \
           zerophase=filter_zerophase_tel, corners=filter_corners_tel)
       times = model.get_travel_times( source_depth_in_km=0.5, distance_in_degree=m_to_deg(dist[i]*1000), \
           phase_list = ['P','p'])
       try:
          arr_P = get_arrival(times, 'P')
       except:
          arr_P = get_arrival(times, 'p')
       if i != 3:
           start_time = UTCDateTime(0) + (arr_P - 0.9*window_bw)
       else:
           start_time = UTCDateTime(0) + (arr_P - 0.55*window_bw)

       end_time = start_time + window_bw
       greens[i]=greens[i].slice(start_time, end_time)
       
       npts = int(window_bw / 0.1 +1)
       array = np.zeros((6, 1, npts))
       ## Rotate to primitive form
       st = greens[i]
       phi = az[i]
       ZSS = st.select(channel='ZSS')[0].data
       ZDS = st.select(channel='ZDS')[0].data
       ZDD = st.select(channel='ZDD')[0].data
       ZEX = st.select(channel='ZEP')[0].data
       array[0, 0, :] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEX/3.
       array[1, 0, :] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEX/3.
       array[2, 0, :] =  ZDD/3. + ZEX/3.
       array[3, 0, :] =  ZSS * np.sin(2*phi)
       array[4, 0, :] =  ZDS * np.cos(phi)
       array[5, 0, :] =  ZDS * np.sin(phi)
       
        
       tele_greens_tensor.append(array)
    
   ## swap axes so Green tensor haveing (elem GF, station, component, time)
   tele_greens_tensor = np.swapaxes(tele_greens_tensor, 0, 1)
   ##
   
   return tele_greens_tensor
