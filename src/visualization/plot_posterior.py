## INITIAL IMPORTATION
# conda -c conda-forge import emcee corner

# %config InlineBackend.figure_format = "retina"
# %load_ext autoreload

from matplotlib import rcParams

rcParams["savefig.dpi"] = 300
rcParams["figure.dpi"] = 300
rcParams["font.size"] = 10

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import numpy as np
import corner
import emcee
from netCDF4 import Dataset
import emcee
from numpy.fft import rfft, irfft, rfftfreq
import os
import glob
import copy
import multiprocessing
import matplotlib
from obspy.geodetics import gps2dist_azimuth
from obspy.imaging.beachball import beach
from pyrocko.plot import beachball
plt.rcParams['savefig.facecolor']='white'

## HELPER FUNCTION TO PLOT LUNE FRAME
from mpl_toolkits.basemap import Basemap
from pyproj import Geod
from pyrocko.moment_tensor import MomentTensor
from mtuq.util.math import to_delta_gamma,to_Mw
from mtuq.misfit.waveform import level2 as level2
from mtuq.event import MomentTensor as MT_mtuq
from src.util.math import to_lune, Tashiro2MT6,Tashiro2MT6_vec,ned2rtp, rtp2ned2, rtp2ned, mt2lune
from src.visualization.lune_diagram import PlotLuneFrame

myfunc = np.vectorize(mt2lune)
g = Geod(ellps='sphere')
bm = Basemap(projection='hammer',lon_0=0)

def posterior_distribution_tt2015(source_type, flat_samples, log_prob, thin, figure_fname):
    MAXVAL = 3600
    if source_type == 'full':
        mt_degree = 6
    elif source_type == 'deviatoric':
        mt_degree = 5
    elif source_type == 'dc':
        mt_degree = 4
    else:
        raise ValueError('wrong source type. It should be full, deviatoric or dc.')
    
    ##v,w,kappa,sigma, h, rho
    m6_samples = flat_samples[::thin,:mt_degree]
    log_prob_2 = log_prob[::thin] / 1e4
    num_samples = len(log_prob_2)
    h_num_samples = int(0.5*num_samples)
    print("number of samples %s and half number is %s " % (num_samples, h_num_samples))
    
    #convert to lune coordinates
    if mt_degree == 6:
        labels = ['Mw', 'Lune-latitude', 'Lune-longitude', 'Strike', 'Rake', 'Dip']
        delta, gamma = to_delta_gamma(m6_samples[:, 0]/10800, m6_samples[:,1]* np.pi / 9600)
        kappa = (m6_samples[:,2]+ MAXVAL) / 20 #strke 0, 360
        sigma = m6_samples[:,3] / 40 #rake -90,90
        dip = np.degrees(np.arccos((m6_samples[:,4]+ MAXVAL) / 7200  )) #0-90
        mw = (m6_samples[:,5]+MAXVAL)/3600 + 4

        thin_sol = 10
        m6_sol = np.array([np.mean(delta[h_num_samples::thin_sol]), np.mean(gamma[h_num_samples::thin_sol]), np.mean(kappa[h_num_samples::thin_sol]), np.mean(sigma[h_num_samples::thin_sol]),np.mean(dip[h_num_samples::thin_sol]), np.mean(mw[h_num_samples::thin_sol])])

        source_samples = np.concatenate((delta[:,np.newaxis], gamma[:,np.newaxis], kappa[:,np.newaxis], \
                                    sigma[:,np.newaxis], dip[:,np.newaxis], mw[:,np.newaxis]), axis=1)
    elif mt_degree==5:#Dev
       labels = ['Mw','Lune_longitude','Strike', 'Rake', 'Dip']
       delta, gamma = to_delta_gamma(m6_samples[:, 0]/10800, 0*m6_samples[:,0])
       kappa = (m6_samples[:,1]+ MAXVAL) / 20 #strke 0, 360
       sigma = m6_samples[:,2] / 40 #rake -90,90
       dip = np.degrees(np.arccos((m6_samples[:,3]+ MAXVAL) / 7200  )) #0-90
       mw = (m6_samples[:,4]+MAXVAL)/3600 + 4
     
       m6_sol = np.array([np.mean(gamma[h_num_samples:]),np.mean(kappa[h_num_samples:]), np.mean(sigma[h_num_samples:]),np.mean(dip[h_num_samples:]), np.mean(mw[h_num_samples:])])
       source_samples = np.concatenate((gamma[:,np.newaxis], kappa[:,np.newaxis], \
                                    sigma[:,np.newaxis], dip[:,np.newaxis], mw[:,np.newaxis]), axis=1)
    elif mt_degree==4:#4 for DC
       labels = ['Mw','Strike', 'Rake', 'Dip']
       kappa = (m6_samples[:,0]+ MAXVAL) / 20 #strke 0, 360
       sigma = m6_samples[:,1] / 40 #rake -90,90
       dip = np.degrees(np.arccos((m6_samples[:,2]+ MAXVAL) / 7200  )) #0-90
       mw = (m6_samples[:,3]+MAXVAL)/3600 + 4
     
       m6_sol = np.array([np.mean(kappa[h_num_samples:]), np.mean(sigma[h_num_samples:]),np.mean(dip[h_num_samples:]), np.mean(mw[h_num_samples:])])
       source_samples = np.concatenate((kappa[:,np.newaxis], \
                                    sigma[:,np.newaxis], dip[:,np.newaxis], mw[:,np.newaxis]), axis=1)
    else:
       raise ValueError("Wrong mt_degree %s" % mt_degree)
     
    print('mean mt:', m6_sol)

    ##setup colour bar
    norm = matplotlib.colors.Normalize(vmin=0.85*max(log_prob_2), vmax=max(log_prob_2))
    cm = copy.copy( plt.get_cmap('copper').reversed())
    # cm.set_under('black')
    ##########

    print(source_samples[h_num_samples:,:].shape)
    #move the Mw to the first column to plot
    source_samples_corner = np.zeros(source_samples.shape)
    source_samples_corner[:,0] = source_samples[:,-1] #Mw
    source_samples_corner[:,1:] = source_samples[:,0:-1]
    m6_sol_corner = np.zeros(6)
    m6_sol_corner[0] = m6_sol[-1]
    m6_sol_corner[1:] = m6_sol[0:-1]
    titles = ['%.2f' % val for val in m6_sol_corner]
    fig = corner.corner(source_samples_corner[h_num_samples::thin_sol,:], labels=labels, truths = m6_sol_corner,titles=titles,title_fmt=None, show_titles=True, truth_color = 'lightcoral',  max_n_ticks=4, label_kwargs={'fontsize':20})

    ##########
    if mt_degree == 4 or mt_degree==5:#DC or Dev
        ax = fig.add_axes([.57, .5, .45, .45])
        ax.set(frame_on=False, xticks=[], yticks=[])
        num_mt = source_samples[h_num_samples::2,:].shape[0]
        print("%s beachballs to plot" % num_mt)
        for i in range(num_mt):
            color = cm(norm(log_prob_2[h_num_samples+i*2]))
            i_dc = source_samples[h_num_samples+i*2]
            if mt_degree==4:
                mt = MomentTensor.from_values([i_dc[0],i_dc[2],i_dc[1],i_dc[3]])
            else:
                mt = MomentTensor.from_values([i_dc[1],i_dc[3],i_dc[2],i_dc[4]])
            b = beach(fm=mt.both_strike_dip_rake()[0], width=0.15*mt.moment_magnitude(),xy=(0.5,0.5), linewidth=1,
                      nofill=True, edgecolor='black', alpha=1)          
            ax.add_collection(b)
        if mt_degree == 4:
            mt = MomentTensor.from_values([m6_sol[0],m6_sol[2],m6_sol[1],m6_sol[3]])
        else:
            mt = MomentTensor.from_values([m6_sol[1],m6_sol[3],m6_sol[2],m6_sol[4]])

        b = beach(fm=mt.both_strike_dip_rake()[0], width=0.15*mt.moment_magnitude(),xy=(0.5,0.5), linewidth=1,
                  nofill=True, edgecolor='lightcoral', alpha=1)          
        ax.add_collection(b)
        plt.annotate('a)', xy=(0.01, 0.99), xycoords='figure fraction', fontweight='bold', va='top', fontsize=20)
        ax.annotate('b)', xy=(0.1, 0.8), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)
    else: #mt_degree =6:
        #plot lune diagram
        #ax1 for the post-burn period
        ax1 = fig.add_axes([.57, .5, .25, .45])
        ax1.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax1, clvd_left=True, clvd_right=False)

        x, y = bm(source_samples[h_num_samples::thin_sol,1], source_samples[h_num_samples::thin_sol,0])
        ax1.scatter(x, y, s=80, c=log_prob_2[h_num_samples::thin_sol], edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(m6_sol[1], m6_sol[0])
        ax1.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')

        #ax2 for the whole chain
        ax2 = fig.add_axes([.75, .5, .25, .45])
        ax2.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax2, clvd_left=False, clvd_right=True)

        x, y = bm(source_samples[:,1], source_samples[:,0])
        ax2.scatter(x, y, s=80, c=log_prob_2, edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(m6_sol[1], m6_sol[0])
        ax2.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax2.legend(loc='lower center', fontsize=12)
        ###

        cax = fig.add_axes([.47, .68, .02, .25])
        cb=matplotlib.colorbar.ColorbarBase(cax,cmap=cm, norm = norm, extend = 'min', orientation='vertical')
        cb.set_label(label='Log_probability / $10^4$', fontsize=20)

        plt.annotate('a)', xy=(0.01, 0.99), xycoords='figure fraction', fontweight='bold', va='top', fontsize=20)
        ax1.annotate('b)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)
        ax2.annotate('c)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)

    plt.savefig(figure_fname)
    
def posterior_distribution_mij(source_type, flat_samples_fname, log_prob_fname, thin, figure_fname):
    ##read the samples and log_likelihood from two .npy files 
    flat_samples = np.load(flat_samples_fname)
    log_prob = np.load(log_prob_fname)
  
    if source_type == 'full':
        mt_degree = 6
    elif source_type == 'deviatoric':
        raise ValueError('Mij method is not implemented for deviatoric MT.')
    else:
        raise ValueError('wrong source type. It should be full or deviatoric.')
    
    ##v,w,kappa,sigma, h, rho
    m6_samples = flat_samples[::thin,:mt_degree]
    log_prob_2 = log_prob[::thin] / 1e4
    num_samples = len(log_prob_2)
    h_num_samples = int(0.5*num_samples)
    print("number of samples %s and half number is %s " % (num_samples, h_num_samples))
    
    #convert to lune coordinates
    if mt_degree == 6:
        labels = ['$M_{xx}$', '$M_{yy}$', '$M_{zz}$', '$M_{xy}$', '$M_{xz}$', '$M_{yz}$']
        source_samples = rtp2ned2(m6_samples[:,0],m6_samples[:,1],m6_samples[:,2], \
                           m6_samples[:,3],m6_samples[:,4],m6_samples[:,5])
        lon,lat = myfunc(source_samples[:,0],source_samples[:,1],source_samples[:,2], \
                         source_samples[:,3],source_samples[:,4],source_samples[:,5])

        m6_sol = np.mean(source_samples[h_num_samples:,], axis=0)
        lon_sol,lat_sol = myfunc(m6_sol[0],m6_sol[1],m6_sol[2], \
                                 m6_sol[3],m6_sol[4],m6_sol[5])
    elif mt_degree==5:#Dev
        pass
    else:
        raise ValueError('wrong mt degree. It should be 5 or 6.')
    print('mean mt:', m6_sol)

    ##setup colar bar
    norm = matplotlib.colors.Normalize(vmin=0.9*max(log_prob_2), vmax=max(log_prob_2))
    cm = copy.copy( plt.get_cmap('copper').reversed())
    # cm.set_under('black')
    ##########

    print(source_samples[h_num_samples:,:].shape)
    titles = ['%.2f' % val for val in m6_sol]
    fig = corner.corner(source_samples[h_num_samples:,:], labels=labels, truths = m6_sol,titles=titles,title_fmt=None, show_titles=True, truth_color = 'lightcoral',  max_n_ticks=4, label_kwargs={'fontsize':20})

    ##########
    if mt_degree == 6:
        #plot lune diagram
        #ax1 for the post-burn period
        ax1 = fig.add_axes([.57, .5, .25, .45])
        ax1.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax1, clvd_left=True, clvd_right=False)

        x, y = bm(lon[h_num_samples:], lat[h_num_samples:])
        ax1.scatter(x, y, s=80, c=log_prob_2[h_num_samples:], edgecolor='none', cmap=cm, norm=norm, alpha=.2)

        x, y = bm(lon_sol, lat_sol)
        ax1.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')

        #ax2 for the whole chain
        ax2 = fig.add_axes([.75, .5, .25, .45])
        ax2.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax2, clvd_left=False, clvd_right=True)

        x, y = bm(lon, lat)
        ax2.scatter(x, y, s=80, c=log_prob_2, edgecolor='none', cmap=cm, norm=norm, alpha=.2)

        x, y = bm(lon_sol, lat_sol)
        ax2.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax2.legend(loc='lower center', fontsize=12)
        ###

        cax = fig.add_axes([.47, .68, .02, .25])
        cb=matplotlib.colorbar.ColorbarBase(cax,cmap=cm, norm = norm, extend = 'min', orientation='vertical')
        cb.set_label(label='Log_probability / $10^4$', fontsize=20)

        plt.annotate('a)', xy=(0.01, 0.99), xycoords='figure fraction', fontweight='bold', va='top', fontsize=20)
        ax1.annotate('b)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)
        ax2.annotate('c)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)

    plt.savefig(figure_fname)

def posterior_distribution_tashiro(source_type, flat_samples, log_prob, thin, figure_fname):
    MAXVAL = 3600
    if source_type == 'full':
        mt_degree = 6
    elif source_type == 'deviatoric':
        raise ValueError('Tashiro method is not implemented for deviatoric MT.')
    else:
        raise ValueError('wrong source type. It should be full or deviatoric.')
    
    ##v,w,kappa,sigma, h, rho
    m6_samples = flat_samples[::thin,:mt_degree]
    log_prob_2 = log_prob[::thin] / 1e4
    num_samples = len(log_prob_2)
    h_num_samples = int(0.5*num_samples)
    print("number of samples %s and half number is %s " % (num_samples, h_num_samples))
    
    #convert to lune coordinates
    if mt_degree == 6:
        labels = ['$X_1$', '$X_2$', '$X_3$', '$X_4$', '$X_5$', '$M_w$']
        m6_samples[:,:5] = (m6_samples[:,:5] + MAXVAL) / 7200
        m6_samples[:,5]  = (m6_samples[:,5] + MAXVAL)/ 3600 + 4
        source_samples = Tashiro2MT6_vec(m6_samples[:,0],m6_samples[:,1],m6_samples[:,2],\
                                         m6_samples[:,3],m6_samples[:,4],m6_samples[:,5])
        lon,lat = myfunc(source_samples[:,0],source_samples[:,1],source_samples[:,2], \
                         source_samples[:,3],source_samples[:,4],source_samples[:,5])

        m6_sol = np.mean(source_samples[h_num_samples:,], axis=0)
        lon_sol,lat_sol = myfunc(m6_sol[0],m6_sol[1],m6_sol[2], \
                                 m6_sol[3],m6_sol[4],m6_sol[5])

    print('mean mt:', m6_sol)

    ##setup colar bar
    norm = matplotlib.colors.Normalize(vmin=0.85*max(log_prob_2), vmax=max(log_prob_2))
    cm = copy.copy( plt.get_cmap('copper').reversed())
    # cm.set_under('black')
    ##########

    print(source_samples[h_num_samples:,:].shape)
    m6_sol = np.mean(m6_samples[h_num_samples::100,:],axis=0)
    titles = ['%.2f' % val for val in m6_sol]
    fig = corner.corner(m6_samples[h_num_samples::100,:], labels=labels, truths =m6_sol ,titles=titles,title_fmt=None, show_titles=True, truth_color = 'lightcoral',  max_n_ticks=4, label_kwargs={'fontsize':20})

    ##########
    if mt_degree == 6:
        #plot lune diagram
        #ax1 for the post-burn period
        ax1 = fig.add_axes([.57, .5, .25, .45])
        ax1.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax1, clvd_left=True, clvd_right=False)

        x, y = bm(lon[h_num_samples::100], lat[h_num_samples::100])
        ax1.scatter(x, y, s=80, c=log_prob_2[h_num_samples::100], edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(lon_sol, lat_sol)
        ax1.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')

        #ax2 for the whole chain
        ax2 = fig.add_axes([.75, .5, .25, .45])
        ax2.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax2, clvd_left=False, clvd_right=True)

        x, y = bm(lon, lat)
        ax2.scatter(x, y, s=80, c=log_prob_2, edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(lon_sol, lat_sol)
        ax2.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax2.legend(loc='lower center', fontsize=12)
        ###

        cax = fig.add_axes([.47, .68, .02, .25])
        cb=matplotlib.colorbar.ColorbarBase(cax,cmap=cm, norm = norm, extend = 'min', orientation='vertical')
        cb.set_label(label='Log_probability / $10^4$', fontsize=20)

        plt.annotate('a)', xy=(0.01, 0.99), xycoords='figure fraction', fontweight='bold', va='top', fontsize=20)
        ax1.annotate('b)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)
        ax2.annotate('c)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)

    plt.savefig(figure_fname)
    
def posterior_distribution_noise(flat_samples_fname, mt_degree, thin, stations, figure_fname):
    rcParams["font.size"] = 20    

    ns = len(stations)
    ##read the samples from a .npy file
    noise = np.load(flat_samples_fname)[::thin,mt_degree:mt_degree+ns]
    
    h_num_samples = int(0.5*noise.shape[0])

    mean_noise = np.mean(noise[h_num_samples:], axis=0)
    print('mean noise:', mean_noise)
    labels = ['$k_{%s}$' % s.station for s in stations]
    titles = ['%.2f' % val for val in mean_noise]
    fig = corner.corner(noise[h_num_samples:,:], labels=labels, truths = mean_noise, titles=titles, title_fmt=None, show_titles=True, \
                        truth_color = 'lightcoral',  max_n_ticks=4, labelpad=0.1, label_kwargs={'fontsize':25,'fontweight':'heavy'})

    plt.savefig(figure_fname, bbox_inches='tight')
    
def posterior_distribution_timeshift(flat_samples_fname, mt_degree, thin, stations, figure_fname):
    rcParams["font.size"] = 20
    
    ns = len(stations)
    ##read the samples from a .npy file
    tau = np.load(flat_samples_fname)[::thin,mt_degree+ns:]
    num_tau = tau.shape[1]

    if num_tau == 2*ns:
        #Rayleigh waves
        tau_Ray = tau[:,::2] 
        #Love waves
        tau_Love = tau[:,1::2] 
    elif num_tau == ns:
        tau_Ray = tau
        tau_Love = tau
    else:
        raise ValueError('wrong number of time shift parameters. It should be ns or 2*ns.')
        
    #Plot time shift for Rayleigh waves
    h_num_samples = int(0.5*tau_Ray.shape[0])

    mean_tau_Ray = np.mean(tau_Ray[h_num_samples:], axis=0)
    print('mean tau for Z/R:', mean_tau_Ray)
    labels = ['$\\tau_{%s}$' % s.station for s in stations]
    titles = ['%.2f' % val for val in mean_tau_Ray]
    fig = corner.corner(tau_Ray[h_num_samples:,:], labels=labels, truths = mean_tau_Ray, titles=titles, title_fmt=None, show_titles=True, \
                        truth_color = 'lightcoral',  max_n_ticks=4, labelpad=0.1, label_kwargs={'fontsize':25,'fontweight':'bold'})
    plt.savefig(figure_fname+'_Rayleigh.jpg', bbox_inches='tight')

    #Plot time shift for Rayleigh waves
    h_num_samples = int(0.5*tau_Ray.shape[0])

    mean_tau_Love = np.mean(tau_Love[h_num_samples:], axis=0)
    print('mean tau for T:', mean_tau_Love)
    titles = ['%.2f' % val for val in mean_tau_Love]
    fig = corner.corner(tau_Love[h_num_samples:,:], labels=labels, truths = mean_tau_Love, titles=titles, title_fmt=None, show_titles=True, \
                        truth_color = 'lightcoral',  max_n_ticks=4, labelpad=0.1, label_kwargs={'fontsize':25,'fontweight':'bold'})
    plt.savefig(figure_fname+'_Love.jpg', bbox_inches='tight')
  
