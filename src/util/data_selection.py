from mtuq import Dataset
import numpy as np
from mtuq.util.math import to_mij, to_rho
from mtuq.grid.moment_tensor import to_mt
from mtuq.grid.force import to_force
from mtuq.grid import UnstructuredGrid
import sys
sys.path.insert(0, '/Users/u7091895/Documents/Research/BayMTI/HiBaysin/src/')
from util.math import to_lune, Tashiro2MT6,ned2rtp

MAXVAL=3600

def rms(data):
    return np.sqrt(np.mean(np.square(data)))
    
def data_noise_estimate_uncorrelated(data_sw, greens_sw, sampling_rate=None):
    #estimate the reference data noise strength for each component
    data_sw_used = []
    greens_sw_used = []
    for i in range(len(data_sw)):
        if len(data_sw[i]) != 0:
            if sampling_rate is not None:
                data_sw[i].resample(sampling_rate=sampling_rate)
                greens_sw[i].resample(sampling_rate=sampling_rate)
            data_sw_used.append(data_sw[i])
            greens_sw_used.append(greens_sw[i])    
    data_sw_used = Dataset(data_sw_used) 
    greens_sw_used = Dataset(greens_sw_used)
    
    ns = len(data_sw_used) #number of stations kept
    nc = 3 # number of components
    components = ['Z','R','T']
    noise_std_sw = np.ones((ns,nc)) 
    for s in range(ns):
       for c in range(nc):
           noise_std_sw[s,c] = rms(data_sw_used[s].select(component=components[c])[0].data) 
           #np.std(data_sw_used[s].select(component=components[c])[0].data, ddof=0) 
   
    return data_sw_used, greens_sw_used, noise_std_sw

def data_noise_estimate_uncorrelated2(data_bw, data_sw, greens_bw, greens_sw, bw_sampling_rate=None, sw_sampling_rate=None):
    #estimate the reference data noise strength for each component
    data_bw_used = []
    data_sw_used = []
    greens_bw_used = []
    greens_sw_used = []
    
    
    for i in range(len(data_sw)):
        ##kept the stations have both body and surface waves
        if len(data_bw[i]) != 0 and len(data_sw[i]) != 0:
            if bw_sampling_rate is not None:
                data_bw[i].resample(sampling_rate=bw_sampling_rate)
                greens_bw[i].resample(sampling_rate=bw_sampling_rate)
            if sw_sampling_rate is not None:
                data_sw[i].resample(sampling_rate=sw_sampling_rate)
                greens_sw[i].resample(sampling_rate=sw_sampling_rate)
                
            data_bw_used.append(data_bw[i])
            data_sw_used.append(data_sw[i])
            greens_bw_used.append(greens_bw[i]) 
            greens_sw_used.append(greens_sw[i])    
    data_bw_used = Dataset(data_bw_used)
    data_sw_used = Dataset(data_sw_used)
    greens_bw_used = Dataset(greens_bw_used)
    greens_sw_used = Dataset(greens_sw_used)
    
    ns = len(data_sw_used) #number of stations kept
    nc = 3 # number of components
    components = ['Z','R','T']
    noise_std_bw = np.ones((ns,nc)) 
    noise_std_sw = np.ones((ns,nc)) 
    for s in range(ns):
       for c in range(nc):
           noise_std_bw[s,c] = np.std(data_bw_used[s].select(component=components[c])[0].data, ddof=0) 
           noise_std_sw[s,c] = np.std(data_sw_used[s].select(component=components[c])[0].data, ddof=0) 
   
    return data_bw_used, data_sw_used, greens_bw_used, greens_sw_used, noise_std_bw, noise_std_sw

def get_solution(emcee_sampler, warm_up_steps, thin, source_type='full'):
        flat_samples = emcee_sampler.get_chain(discard=warm_up_steps, thin=thin, flat=True)
        print ('\nNumber of quasi-independent samples: %d' % flat_samples.shape[0])
        m_sol = np.mean(flat_samples, axis=0)
 
        ##transformed MT parameters from Lune to premitive parameterization
        if source_type=='full':
            v = m_sol[0] / 10800
            w = m_sol[1] * np.pi / 9600             
            kappa = (m_sol[2] + MAXVAL) / 20           ##0, 360
            sigma = m_sol[3] / 40                      ##-90, 90
            h = (m_sol[4] + MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[5]+MAXVAL)/3600 + 4) ##Mw: 4-6
        elif source_type=='dc':
            v = 0
            w = 0            
            kappa = (m_sol[0] + MAXVAL) / 20           ##0, 360
            sigma = m_sol[1] / 40                      ##-90, 90
            h = (m_sol[2] + MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[3]+MAXVAL)/3600 + 4) ##Mw: 4-6
        elif source_type=='deviatoric':
            v = m_sol[0] / 10800
            w = 0          
            kappa = (m_sol[1] + MAXVAL) / 20           ##0, 360
            sigma = m_sol[2] / 40                      ##-90, 90
            h = (m_sol[3] + MAXVAL) / 7200             ##cos(dip)
            rho = to_rho((m_sol[4]+MAXVAL)/3600 + 4) ##Mw: 4-6
        elif source_type=='force':
            phi = (m_sol[0]+ MAXVAL) / 20   #[0, 360]
            h = m_sol[1] / 3600             #[-1,1]
            F0 = (m_sol[2] + MAXVAL)        #[0,7200]
        elif source_type == 'mij':
            m_sol *= 10**15
            rho,v,w,kappa,sigma,h = to_lune(m_sol[0:6])
        else:
            #source type = Tashiro
            m_sol[:5] = (MAXVAL + m_sol[:5]) / 7200   #(0,1)
            m_sol[5] = (MAXVAL + m_sol[5]) / 3600 + 4 #Mw 4-6
            mij = Tashiro2MT6(m_sol[:6])
            mij = ned2rtp(mij) #up-south-east convention
            rho,v,w,kappa,sigma,h = to_lune(mij)
            
        if source_type=='force':
            solution = UnstructuredGrid(
                dims=('F0','phi', 'h'),
                coords=(F0, phi, h),
                callback=to_force)   
        else:
            ## moment tensor
            solution = UnstructuredGrid(
                dims=('rho','v', 'w', 'kappa', 'sigma', 'h'),
                coords=(rho, v,w,kappa,sigma,h),
                callback=to_mt)
        return solution