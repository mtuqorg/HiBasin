from telnetlib import EC

import numpy as np
from os.path import exists, abspath
from os import makedirs, chdir, getcwd, system
from obspy.core import Stream, Trace, UTCDateTime
from obspy.geodetics import gps2dist_azimuth
import os

ECOMPS = ['ZDD', 'RDD', 'ZDS', 'RDS', 'TDS', 'ZSS', 'RSS', 'TSS', 'ZEX', 'REX']

class File96:
    def __init__(self):
        ## Format edition: FILE01.02, FILE03.02, FILE16.02
        self.format = ''
        ## Types of waveforms: OBSERVED, SYNTHETIC
        self.type = ''
        ## Time or frequency domains: TIME_DOMAIN, FREQUENCY_DOMAIN
        self.domain = 'TIME_DOMAIN'
        ## Ground motion units
        self.unit = ''
        ## Processing of the waveforms
        self.processing = 'NONE'
        ## Event origin time and location
        self.orig_time = UTCDateTime(1970, 1, 1, 0, 0, 0)
        self.orig_lat = 0
        self.orig_lon = 0
        self.orig_dep = 0
        ## Station name, 'GRN??' if sythetic
        self.station = ''
        ## Station location
        self.stat_lat = 0
        self.stat_lon = 0
        self.stat_el  = 0
        ## Event-station geometry
        self.dist_km  = 0
        self.dist_deg = 0
        self.az = 0
        self.baz = 0
        ## Source pulse applied to synthetic waveforms
        self.source_pulse = ''
        ## Comment line/velocity model if synthetic
        self.comment = ''
        ## Pressure unit
        self.pressure_unit = ''
        ## Prediction of P, SV, SH arrivals
        self.pred_P = 0
        self.pred_SV = 0
        self.pred_SH = 0
        ## Medium parameters at the source depth: A, C, F, L, N, rho
        self.medium_params = np.zeros(6)
        ## Shared common properties between component traces
        self.ncomps = 21 # number of components
        self.npts = 0    # number of points in each trace
        self.offset = 0  # trace starting time offset with origin time      
        self.delta = 0   # data trace time step
        ## Indication of traces with active data
        self.JSRC = []
        ## Component names
        self.comp = []
        ## Component incidences with respect to the vertical
        self.comp_inc = []
        ## Component azimuths
        self.comp_azm = []
        ## Component data traces
        self.data = None

def readFile96(fname):
    ## Load file96 content
    content = open(fname, 'r').readlines()
    ## Initialize file object
    f96 = File96()
    ## Format
    f96.format = content[0].strip()
    ## Type
    f96.type = content[1].strip()
    ## Domain 
    f96.domain = content[2].strip()
    ## Unit
    f96.unit = content[3].strip()
    ## Processing
    f96.processing = content[4].strip()
    ## Event origin
    tmp = content[5].split()
    if (f96.type != 'SYNTHETIC'):
        f96.orig_time = UTCDateTime(int(tmp[0]), int(tmp[1]), int(tmp[2]), \
            int(tmp[3]), int(tmp[4]), float(tmp[5]))
    f96.evla = float(tmp[6])
    f96.evlo = float(tmp[7])
    f96.evdp = float(tmp[8])
    ## Station name
    f96.station = content[6].strip()
    ## Station location
    tmp = content[7].split()
    f96.stla = float(tmp[0])
    f96.stlo = float(tmp[1])
    f96.stel = float(tmp[2])
    ## Source-station geometry
    tmp = content[8].split()
    f96.dist_km = float(tmp[0])
    f96.dist_deg = float(tmp[1])
    f96.az = float(tmp[2])
    f96.baz = float(tmp[3])
    ## Source pulse manipulation
    f96.source_pulse = content[9].strip()
    ## Comment line
    f96.comment = content[10].strip()
    ## Pressure unit
    f96.pressure_unit = content[11].strip()
    ## Prediction arrivals
    tmp = content[12].split()
    f96.pred_P  = float(tmp[0])
    f96.pred_SV = float(tmp[1])
    f96.pred_SH = float(tmp[2])
    ## Medium parameters at source depth
    tmp = np.array(content[13].split())
    f96.medium_params = tmp.astype(float)
    ## Indication of traces with active data
    tmp = np.array(content[16].split())
    f96.JSRC = tmp.astype(int)
    f96.JSRC = f96.JSRC[f96.JSRC>0]
    ## Number of columns 
    f96.ncomps = len(f96.JSRC)
    ## Trace delta 
    tmp = content[18].split()
    f96.delta = float(tmp[2])
    f96.npts = int(tmp[3])
    if f96.type == 'SYNTHETIC':
        tmp = content[19].split()
        f96.offset = float(tmp[5])
    ## Read actual waveform data
    f96.data = np.zeros((f96.ncomps, f96.npts))
    f96.comp = [''] * f96.ncomps
    f96.comp_inc = np.zeros(f96.ncomps)
    f96.comp_azm = np.zeros(f96.ncomps)
    ## Determine number of data lines given that a line has 4 data points
    num_datalines = int(np.ceil(f96.npts / 4))
    ## Iterate through component items
    component_startline = 17
    for nc in range(f96.ncomps):
        ## If there is no active data, skip the trace
        if f96.JSRC[nc] == 0: continue
        ## Component name
        f96.comp[nc] = content[component_startline].strip()
        ## Component angle of incidence and azimuth
        tmp = content[component_startline+1].split()
        f96.comp_inc[nc] = float(tmp[0])
        f96.comp_azm[nc] = float(tmp[1])
        ## Read through data lines
        tmp_data = []
        for i in range(num_datalines):
            tmp_data.extend(content[component_startline+3+i].split())
        tmp_data = np.array(tmp_data).astype(float)
        component_startline += (3 + num_datalines)
        ## Copy data
        f96.data[nc, :] = tmp_data[0:f96.npts] 
    ## Convert Green's function generated by moment 1e20 dyne to m/s
    f96.data /= 1e15
    ## Return complete object
    return f96

def writeModel96(vel_model, fname):
    '''
    This function write a 1D model, including specification of layer thicknesses,
    P-, S-wave velocities and densities, to file in Model96 format (used in CPS).
    '''
    thick = vel_model[0, :]
    vp = vel_model[1, :]
    vs = vel_model[2, :]
    rho = vel_model[3, :]
    qka = vel_model[4, :]
    qmu = vel_model[5, :]

    fid = open(fname, 'w')
    fid.write('MODEL.01\n' +
            'Korean Pennisula model from Kim et al. (2011)\n' +
            'ISOTROPIC\n' +
            'KGS\n' +
            'FLAT EARTH\n' +
            '1-D\n' +
            'CONSTANT VELOCITY\n' +
            'LINE08\n' +
            'LINE09\n' +
            'LINE10\n' +
            'LINE11\n')
    fid.write('H(KM)    VP(KM/S) VS(KM/S) RHO(GM/CC)  QP       QS    ETAP     ETAS   FREFP    FREFS\n')
    for n in range(len(thick)):
        line = '%-8.2f %-8.3f %-8.3f %-8.3f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f %-8.1f\n' % \
            (thick[n], vp[n], vs[n], rho[n], qka[n], qmu[n], 0, 0, 1, 1)
        fid.write(line)
    fid.close()

def generate_CPS_greens_tensor(origin, stations, path_to_earth_model, dt=0.5, npts=2048, t0=0, vred=0):
    ## Create a directory to store generated Green's function
    cps_dname = '%s_%dm' % (path_to_earth_model, origin.depth_in_m)
    if not exists(cps_dname): makedirs(cps_dname)

    print ('    Generating Greens\'s function with CPS\n')

    # create dfile
    fid = open('%s/dfile' % cps_dname, 'w')
    for s in stations:
        distance_in_m, _, _ = gps2dist_azimuth(origin.latitude, origin.longitude, s.latitude, s.longitude)
        fid.write('%.1f %.2f %d %.1f %.1f\n' % (distance_in_m/1000, dt, npts, t0, vred))
    fid.close()

    # write mod96 velocity model
    mod_fname = abspath('%s.mod' % path_to_earth_model)
    mod_fname = '/Users/u7091895/Documents/Research/BayMTI/HiBaysin/examples/MDJ2.mod'
    print(mod_fname)
    #writeModel96(np.loadtxt(path_to_earth_model).T, mod_fname)

    current_dname = getcwd()
    chdir(cps_dname)
    ## main jobs done with CPS routines
    system('hprep96 -M %s -d dfile -HS %s -HR 0.0 -EQEX -R' % (mod_fname, origin.depth_in_m/1e3))
    system('hspec96 > hspec96.out')
    system('hpulse96 -D -i > hpulse96.out')

    ## split file96 data into multiple files corresponding to each station
    for i, s in enumerate(stations):
        system('cat hpulse96.out | fsel96 -NS %d > %s_elemGF.f96' % (i+1, s.station))
    ##
    #system('rm -f hpulse96.out hspec96.dat  hspec96.grn  hspec96.out')
    chdir(current_dname)

def generate_CPS_greens_tensor_2D(origin, stations, path_to_earth_model, dt=0.5, npts=2048, t0=0, vred=0):
    ## Create a directory to store generated Green's function
    cps_dname = '%s_%dm' % (path_to_earth_model, origin.depth_in_m)
    if not exists(cps_dname): makedirs(cps_dname)

    print ('    Generating Greens\'s function with CPS\n')
    current_dname = getcwd()
    for s in stations:
        # create dfile
        fid = open('%s/%s_dfile' % (cps_dname, s.station), 'w')
        distance_in_m, _, _ = gps2dist_azimuth(origin.latitude, origin.longitude, s.latitude, s.longitude)
        fid.write('%.1f %.2f %d %.1f %.1f\n' % (distance_in_m/1000, dt, npts, t0, vred))
        fid.close()

        # get the name of mod96 velocity model for the station
        # mod_fname = abspath('%s.mod' % path_to_earth_model)
        mod_fname = path_to_earth_model + '/%s.mod' % s.station
        print(mod_fname)
            
        chdir(cps_dname)
        ## main jobs done with CPS routines
        system('hprep96 -M %s -d %s_dfile -HS %s -HR 0.0 -EQEX -R' % (mod_fname, s.station, origin.depth_in_m/1e3))
        system('hspec96 > hspec96.out')
        system('hpulse96 -D -p -l 1  > hpulse96.out')
        #system('hpulse96 -D -i  > hpulse96.out')
    
        ## change the name to station
        system('cat hpulse96.out | fsel96 -NS 1 > %s_elemGF.f96' % s.station)
        system('f96tosac -G <%s_elemGF.f96' % s.station)
        ##
        system('rm -f hpulse96.out hspec96.dat  hspec96.grn  hspec96.out')
    chdir(current_dname)
    
def load_CPS_greens_tensor(origin, stations, path_to_data, dt=1.0, npts=256, \
    filter_dict={'fmin':0.02, 'fmax':0.05, 'corners':4, 'zerophase':True}):
    greens_tensor = []
    ne = 6
    nc = 3

    for s in stations:
        _, az, _ = gps2dist_azimuth(origin.latitude, origin.longitude, s.latitude, s.longitude)
        phi = np.radians(az)
        f96 = readFile96(os.path.join(path_to_data, '%s_elemGF.f96' % s.station))
        ## Preprocess calculated Green's tensor
        st = Stream()
        for e, comp in enumerate(ECOMPS):
             st.append(Trace(header={'delta':f96.delta, 'channel':comp}, data=f96.data[e]))
        st.filter('bandpass', freqmin=filter_dict['fmin'], freqmax=filter_dict['fmax'],
            corners=filter_dict['corners'], zerophase=filter_dict['zerophase'])
        st.resample(1/dt)
        # st.trim(st[0].stats.starttime, st[0].stats.starttime+npts*dt)

        array = np.zeros((ne, nc, npts))
        ## Rotate to primitive form
        ZSS = st.select(channel='ZSS')[0].data[0:npts]
        ZDS = st.select(channel='ZDS')[0].data[0:npts]
        ZDD = st.select(channel='ZDD')[0].data[0:npts]
        ZEX = st.select(channel='ZEX')[0].data[0:npts]
        array[0, 0, :] =  ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEX/3.
        array[1, 0, :] = -ZSS/2. * np.cos(2*phi) - ZDD/6. + ZEX/3.
        array[2, 0, :] =  ZDD/3. + ZEX/3.
        array[3, 0, :] =  ZSS * np.sin(2*phi)
        array[4, 0, :] =  ZDS * np.cos(phi)
        array[5, 0, :] =  ZDS * np.sin(phi)
        RSS = st.select(channel='RSS')[0].data[0:npts]
        RDS = st.select(channel='RDS')[0].data[0:npts]
        RDD = st.select(channel='RDD')[0].data[0:npts]
        REX = st.select(channel='REX')[0].data[0:npts]
        array[0, 1, :] =  RSS/2. * np.cos(2*phi) - RDD/6. + REX/3.
        array[1, 1, :] = -RSS/2. * np.cos(2*phi) - RDD/6. + REX/3.
        array[2, 1, :] =  RDD/3. + REX/3.
        array[3, 1, :] =  RSS * np.sin(2*phi)
        array[4, 1, :] =  RDS * np.cos(phi)
        array[5, 1, :] =  RDS * np.sin(phi)
        TSS = st.select(channel='TSS')[0].data[0:npts]
        TDS = st.select(channel='TDS')[0].data[0:npts]
        array[0, 2, :] =  TSS/2. * np.sin(2*phi)
        array[1, 2, :] = -TSS/2. * np.sin(2*phi)
        array[2, 2, :] =  0.
        array[3, 2, :] = -TSS * np.cos(2*phi)
        array[4, 2, :] =  TDS * np.sin(phi)
        array[5, 2, :] = -TDS * np.cos(phi)

        greens_tensor.append(array)
    
    ## swap axes so Green tensor haveing (elem GF, station, component, time)
    greens_tensor = np.swapaxes(greens_tensor, 0, 1)
    return greens_tensor

from IO import read_stations, read_event
if __name__ == '__main__':
    evname = 'NK2017'
    stations = read_stations('%s.config' % evname)
    origin = read_event()

    # generate_CPS_greens_tensor(origin, stations, 'CPS/mdj2')
    load_CPS_greens_tensor(origin, stations, 'CPS/mdj2_1000m')
