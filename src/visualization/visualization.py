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
import sys
sys.path.insert(0, '/Users/u7091895/Documents/Research/BayMTI/HiBaysin/src/')
from util.math import to_lune, Tashiro2MT6,Tashiro2MT6_vec,ned2rtp, rtp2ned2, rtp2ned

g = Geod(ellps='sphere')
bm = Basemap(projection='hammer',lon_0=0)

def PlotLuneFrame(ax, fc='white', clvd_left=True, clvd_right=True):
    ## Make sure that the axis has equal aspect ratio
    ax.set_aspect('equal')
    ## Plot meridian grid lines
    lats = np.arange(-90, 91)
    for lo in range(-30, 31, 10):
        lons = np.ones(len(lats)) * lo
        x, y = bm(lons, lats)
        ax.plot(x, y, lw=0.5, color='gray')
    x0, y = bm(-30*np.ones(len(lats)), lats)
    x1, y = bm(30*np.ones(len(lats)), lats)
    ax.fill_betweenx(y, x0, x1, color=fc, lw=0)
    ## Plot the left most meridian boundary
    lons = np.ones(len(lats)) * -30
    x, y = bm(lons, lats)
    ax.plot(x, y, lw=1, color='k')
    ## Plot the right most meridian boundary
    lons = np.ones(len(lats)) * 30
    x, y = bm(lons, lats)
    ax.plot(x, y, lw=1, color='k')
    ## Plot parallel grid lines
    lons = np.arange(-30, 31)
    for la in range(-90, 91, 10):
        lats = np.ones(len(lons)) * la
        x, y = bm(lons, lats)
        ax.plot(x, y, lw=0.5, color='gray')
    ## Put markers on special mechanism
    #-- isotropic points
    x, y = bm(0, 90)
    ax.plot(x, y, 'ok', markersize=2)
    ax.annotate('+ISO', xy=(x, y*1.03), fontsize=15, fontweight='bold', ha='center')
    ax.plot(x, 0, 'ok', markersize=2)
    ax.annotate('-ISO', xy=(x, -y*.03), fontsize=15, fontweight='bold', ha='center', va='top')
    #-- CLVD points
    x, y = bm(30, 0)
    ax.plot(x, y, 'ok', markersize=2)
    if clvd_right: ax.annotate('-CLVD', xy=(1., 0.5), xycoords='axes fraction', fontsize=15, fontweight='bold', \
                rotation='vertical', va='center')
    x, y = bm(-30, 0)
    ax.plot(x, y, 'ok', markersize=2)
    if clvd_left: ax.annotate('+CLVD', xy=(0, 0.5), xycoords='axes fraction', fontsize=15, fontweight='bold', 
                rotation='vertical', ha='right', va='center')
    # -- Double couple point
    x, y = bm(0, 0)
    ax.plot(x, y, 'ok', markersize=2)
    # ax.annotate('DC', xy=(x, y*1.03), fontweight='bold', \
    #             ha='center', va='bottom')
    # -- LVD
    lvd_lon = 30
    lvd_lat = np.degrees(np.arcsin(1/np.sqrt(3)))
    x, y = bm(-lvd_lon, lvd_lat)
    ax.plot(x, y, 'ok', markersize=2)
    x, y = bm(lvd_lon, 90-lvd_lat)
    ax.plot(x, y, 'ok', markersize=2)
    arc = g.npts(-lvd_lon, lvd_lat, lvd_lon, 90-lvd_lat, 50)
    x, y = bm([p[0] for p in arc], [p[1] for p in arc])
    ax.plot(x, y, lw=1, color='k')

    x, y = bm(-lvd_lon, lvd_lat-90)
    ax.plot(x, y, 'ok', markersize=2)
    x, y = bm(lvd_lon, -lvd_lat)
    ax.plot(x, y, 'ok', markersize=2)
    arc = g.npts(-lvd_lon, lvd_lat-90, lvd_lon, -lvd_lat, 50)
    x, y = bm([p[0] for p in arc], [p[1] for p in arc])
    ax.plot(x, y, lw=1, color='k')


def MT6toMT9(mt):
    mt9 = np.zeros((3,3))
    mt9[0,:] = np.array([mt[0], mt[3], mt[4]])
    mt9[1,:] = np.array([mt[3], mt[1], mt[5]])
    mt9[2,:] = np.array([mt[4], mt[5], mt[2]])
    return mt9
    
def MT9toNatural(mt9):
    pyrocko_mt = MomentTensor(m=mt9)
    eigenvals = pyrocko_mt.eigenvals()
    eigenvals.sort()
    lambda1, lambda2, lambda3 = eigenvals[::-1] # lamda1 >= lambda2 >= lambda3
    ## Coordinates of source-type lune diagram
    rho = np.sqrt(np.sum(eigenvals.dot(eigenvals)))
    gamma = np.arctan2((-lambda1+2*lambda2-lambda3), (np.sqrt(3)*(lambda1-lambda3))) # longitude
    beta= np.arccos((lambda1+lambda2+lambda3) / (np.sqrt(3)*rho))
    sigma = np.pi/2 - beta # latitude
    strike, dip, rake = pyrocko_mt.both_strike_dip_rake()[0]
    return (np.degrees(gamma), np.degrees(sigma), strike, dip, rake, pyrocko_mt.moment_magnitude())

## CORNER PLOT SHOWING TRADE-OFF IN MT COMPONENTS AND SOURCE TYPES ON LUNE
def mt2lune(mxx, myy, mzz, mxy, mxz, myz):
    m33 = np.array([[mxx, mxy, mxz], [mxy, myy, myz], [mxz, myz, mzz]])
    eivals = np.linalg.eigvals(m33)
    eivals.sort()
    
    ## lune longitude calculated from the eigen value triple
    nom = -eivals[0] - eivals[2] + 2 * eivals[1]
    den = np.sqrt(3) * (eivals[2]- eivals[0])
    gamma = np.arctan2(nom, den) / np.pi * 180

    ## lune latitude calculated from the eigen value triple
    nom = np.sum(eivals)
    den = np.sqrt(3) * np.sqrt(np.sum(eivals**2))
    beta = np.arccos(nom / den) / np.pi * 180

    ## orientation angles determined from the eigen vector triple
    return gamma, 90 - beta

myfunc = np.vectorize(mt2lune)

def posterior_distribution(source_type, flat_samples, log_prob, thin, figure_fname):
    MAXVAL = 3600
    if source_type == 'full':
        mt_degree = 6
    elif source_type == 'deviatoric':
        mt_degree = 5
    elif source_type == 'dc':
        mt_degree = 4
    elif source_type == 'force':
        mt_degree = 3
    else:
        print('wrong source type.')
        exit(0)
    
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
    else:#force
       labels = ['Phi', 'Theta', 'F0']
       phi = (m6_samples[:,0]+ MAXVAL) / 20 # 0, 360
       theta = np.degrees(np.arccos(m6_samples[:,1] / 3600)) #-+90
       F0 = (m6_samples[:,2]+ MAXVAL)
     
       m6_sol = np.array([np.mean(phi[h_num_samples:]), np.mean(theta[h_num_samples:]),np.mean(F0[h_num_samples:])])
       source_samples = np.concatenate((phi[:,np.newaxis], theta[:,np.newaxis], F0[:,np.newaxis]), axis=1)
    print('mean mt:', m6_sol)

    ##setup colar bar
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
    if mt_degree == 4 or mt_degree==5:#DC
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
    if mt_degree > 5:
        ##only for synthetic experiment
        # mt_true = np.loadtxt('mt_input.txt')
        #DC
#         mt_true = np.array([4.66977228e+16, -3.78098910e+16, -8.88783183e+15, -1.47054371e+16,
#                               3.71126807e+15, -2.09101333e+16])
        #CLVD
        mt_true = np.array([2.760, -1.661, -1.099, 0.707, 1.048, 0.634])*3.0e16
        lon, lat = mt2lune(mt_true[0],mt_true[1],mt_true[2],mt_true[3],mt_true[4],mt_true[5])
        x0, y0 = bm(lon,lat)
        
        #plot lune diagram
        #ax1 for the post-burn period
        ax1 = fig.add_axes([.57, .5, .25, .45])
        ax1.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax1, clvd_left=True, clvd_right=False)

        x, y = bm(source_samples[h_num_samples::thin_sol,1], source_samples[h_num_samples::thin_sol,0])
        ax1.scatter(x, y, s=80, c=log_prob_2[h_num_samples::thin_sol], edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(m6_sol[1], m6_sol[0])
        ax1.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax1.scatter(x0, y0, s=190, c='none', edgecolor='cyan', label='True MT', marker='*')

        #ax2 for the whole chain
        ax2 = fig.add_axes([.75, .5, .25, .45])
        ax2.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax2, clvd_left=False, clvd_right=True)

        x, y = bm(source_samples[:,1], source_samples[:,0])
        ax2.scatter(x, y, s=80, c=log_prob_2, edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(m6_sol[1], m6_sol[0])
        ax2.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax2.scatter(x0, y0, s=190, c='none', edgecolor='cyan', label='True MT', marker='*')
        ax2.legend(loc='lower center', fontsize=12)
        ###

        cax = fig.add_axes([.47, .68, .02, .25])
        cb=matplotlib.colorbar.ColorbarBase(cax,cmap=cm, norm = norm, extend = 'min', orientation='vertical')
        cb.set_label(label='Log_probability / $10^4$', fontsize=20)

        plt.annotate('a)', xy=(0.01, 0.99), xycoords='figure fraction', fontweight='bold', va='top', fontsize=20)
        ax1.annotate('b)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)
        ax2.annotate('c)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)

    plt.savefig(figure_fname)
    
def posterior_distribution_mij(source_type, flat_samples, log_prob, thin, figure_fname):
    MAXVAL = 3600
    if source_type == 'full':
        mt_degree = 6
    elif source_type == 'deviatoric':
        mt_degree = 5
    elif source_type == 'force':
        mt_degree = 3
    else:
        print('wrong source type.')
        exit(0)
    
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
    else:#force
        labels = ['Phi', 'Theta', 'F0']
        phi = (m6_samples[:,0]+ MAXVAL) / 20 # 0, 360
        theta = np.degrees(np.arccos(m6_samples[:,1] / 3600)) #-+90
        F0 = (m6_samples[:,2]+ MAXVAL)
     
        m6_sol = np.array([np.mean(phi[h_num_samples:]), np.mean(theta[h_num_samples:]),np.mean(F0[h_num_samples:])])
        source_samples = np.concatenate((phi[:,np.newaxis], theta[:,np.newaxis], F0[:,np.newaxis]), axis=1)
    print('mean mt:', m6_sol)

    ##setup colar bar
    norm = matplotlib.colors.Normalize(vmin=0.85*max(log_prob_2), vmax=max(log_prob_2))
    cm = copy.copy( plt.get_cmap('copper').reversed())
    # cm.set_under('black')
    ##########

    print(source_samples[h_num_samples:,:].shape)
    titles = ['%.2f' % val for val in m6_sol]
    fig = corner.corner(source_samples[h_num_samples:,:], labels=labels, truths = m6_sol,titles=titles,title_fmt=None, show_titles=True, truth_color = 'lightcoral',  max_n_ticks=4, label_kwargs={'fontsize':20})

    ##########
    if mt_degree > 5:
        ##only for synthetic experiment
        # mt_true = np.loadtxt('mt_input.txt')
        #DC
        mt_true = np.array([4.66977228e+16, -3.78098910e+16, -8.88783183e+15, -1.47054371e+16,
                              3.71126807e+15, -2.09101333e+16])
        #CLVD
        # mt_true = np.array([2.760, -1.661, -1.099, 0.707, 1.048, 0.634])*3.0e16
        lon_true, lat_true = mt2lune(mt_true[0],mt_true[1],mt_true[2],mt_true[3],mt_true[4],mt_true[5])
        x0, y0 = bm(lon_true,lat_true)
        
        #plot lune diagram
        #ax1 for the post-burn period
        ax1 = fig.add_axes([.57, .5, .25, .45])
        ax1.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax1, clvd_left=True, clvd_right=False)

        x, y = bm(lon[h_num_samples:], lat[h_num_samples:])
        ax1.scatter(x, y, s=80, c=log_prob_2[h_num_samples:], edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(lon_sol, lat_sol)
        ax1.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax1.scatter(x0, y0, s=190, c='none', edgecolor='cyan', label='True MT', marker='*')

        #ax2 for the whole chain
        ax2 = fig.add_axes([.75, .5, .25, .45])
        ax2.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax2, clvd_left=False, clvd_right=True)

        x, y = bm(lon, lat)
        ax2.scatter(x, y, s=80, c=log_prob_2, edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(lon_sol, lat_sol)
        ax2.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax2.scatter(x0, y0, s=190, c='none', edgecolor='cyan', label='True MT', marker='*')
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
        mt_degree = 5
    elif source_type == 'force':
        mt_degree = 3
    else:
        print('wrong source type.')
        exit(0)
    
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

       
    elif mt_degree==5:#Dev
        pass
    else:#force
        labels = ['Phi', 'Theta', 'F0']
        phi = (m6_samples[:,0]+ MAXVAL) / 20 # 0, 360
        theta = np.degrees(np.arccos(m6_samples[:,1] / 3600)) #-+90
        F0 = (m6_samples[:,2]+ MAXVAL)
     
        m6_sol = np.array([np.mean(phi[h_num_samples:]), np.mean(theta[h_num_samples:]),np.mean(F0[h_num_samples:])])
        source_samples = np.concatenate((phi[:,np.newaxis], theta[:,np.newaxis], F0[:,np.newaxis]), axis=1)
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
    if mt_degree > 5:
        ##only for synthetic experiment
        mt_true = np.loadtxt('mt_input.txt')
        #DC
#         mt_true = np.array([4.66977228e+16, -3.78098910e+16, -8.88783183e+15, -1.47054371e+16,
#                               3.71126807e+15, -2.09101333e+16])
        #CLVD
#         mt_true = np.array([2.760, -1.661, -1.099, 0.707, 1.048, 0.634])*3.0e16
        lon_true, lat_true = mt2lune(mt_true[0],mt_true[1],mt_true[2],mt_true[3],mt_true[4],mt_true[5])
        x0, y0 = bm(lon_true,lat_true)
        
        #plot lune diagram
        #ax1 for the post-burn period
        ax1 = fig.add_axes([.57, .5, .25, .45])
        ax1.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax1, clvd_left=True, clvd_right=False)

        x, y = bm(lon[h_num_samples::100], lat[h_num_samples::100])
        ax1.scatter(x, y, s=80, c=log_prob_2[h_num_samples::100], edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(lon_sol, lat_sol)
        ax1.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax1.scatter(x0, y0, s=190, c='none', edgecolor='cyan', label='True MT', marker='*')

        #ax2 for the whole chain
        ax2 = fig.add_axes([.75, .5, .25, .45])
        ax2.set(frame_on=False, xticks=[], yticks=[])
        PlotLuneFrame(ax2, clvd_left=False, clvd_right=True)

        x, y = bm(lon, lat)
        ax2.scatter(x, y, s=80, c=log_prob_2, edgecolor='none', cmap=cm, norm=norm, alpha=.1)

        x, y = bm(lon_sol, lat_sol)
        ax2.scatter(x, y, s=190, c='red', edgecolor='red', label='Mean MT solution', marker='+')
        ax2.scatter(x0, y0, s=190, c='none', edgecolor='cyan', label='True MT', marker='*')
        ax2.legend(loc='lower center', fontsize=12)
        ###

        cax = fig.add_axes([.47, .68, .02, .25])
        cb=matplotlib.colorbar.ColorbarBase(cax,cmap=cm, norm = norm, extend = 'min', orientation='vertical')
        cb.set_label(label='Log_probability / $10^4$', fontsize=20)

        plt.annotate('a)', xy=(0.01, 0.99), xycoords='figure fraction', fontweight='bold', va='top', fontsize=20)
        ax1.annotate('b)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)
        ax2.annotate('c)', xy=(0.01, 0.99), xycoords='axes fraction', fontweight='bold', va='top', fontsize=20)

    plt.savefig(figure_fname)
    
def posterior_distribution_noise(flat_samples, mt_degree,ns, thin, stations, figure_fname):
    rcParams["font.size"] = 20
    MAXVAL = 3600
    
    #convert to transformed parameters
    noise = 2*(flat_samples[::thin,mt_degree:mt_degree+ns] + MAXVAL)/720 + 0.0001
    h_num_samples = int(0.5*noise.shape[0])

    mean_noise = np.mean(noise[h_num_samples:], axis=0)
    print('mean noise:', mean_noise)
    labels = ['$k_{%s}$' % s for s in stations]
    titles = ['%.2f' % val for val in mean_noise]
    fig = corner.corner(noise[h_num_samples:,:], labels=labels, truths = mean_noise, titles=titles, title_fmt=None, show_titles=True, truth_color = 'lightcoral',  max_n_ticks=4, labelpad=0.1, label_kwargs={'fontsize':25,'fontweight':'heavy'})

    plt.savefig(figure_fname, bbox_inches='tight')
    
def posterior_distribution_timeshift(flat_samples, mt_degree, ns, thin, stations, figure_fname, wave_type):
    rcParams["font.size"] = 20
    MAXVAL = 3600
    
    #convert to transformed parameters
    if wave_type=='R':
        tau = flat_samples[::thin,mt_degree+ns::2] / 360
    else:
        tau = flat_samples[::thin,mt_degree+ns+1::2] / 360      
          
    h_num_samples = int(0.5*tau.shape[0])

    mean_tau = np.mean(tau[h_num_samples:], axis=0)
    print('mean tau:', mean_tau)
    labels = ['$\\tau_{%s}$' % s for s in stations]
    titles = ['%.2f' % val for val in mean_tau]
    fig = corner.corner(tau[h_num_samples:,:], labels=labels, truths = mean_tau, titles=titles, title_fmt=None, show_titles=True, truth_color = 'lightcoral',  max_n_ticks=4, labelpad=0.1, label_kwargs={'fontsize':25,'fontweight':'bold'})


    plt.savefig(figure_fname, bbox_inches='tight')
  

def plot_waveform_fit(mij_sol, data, greens, stations, noise, tau, figure_fname, evdp_in_km=33):

    ##data preparation
#     nt, delta = level2._get_time_sampling(data_sw)
#     stations = level2._get_stations(data_sw)
#     components = level2._get_components(data_sw)

#     #
#     # collapse main structures into NumPy arrays
#     #
#     data = level2._get_data(data_sw, stations, components)       #ns x nc x nt
#     greens = level2._get_greens(greens_sw, stations, components) #ns x nc x ne x nt in up-south-east convention
    
    ns,nc,ne,nt = greens.shape
    print(data.shape, greens.shape)
    ###

    pred = np.einsum('scet,e->sct', greens, mij_sol)
    delta =1

    omega = 2*np.pi * rfftfreq(nt, d=delta)
    shifted_pred = np.zeros(pred.shape)
    for s in range(ns):
        shifted_pred[s,:2] = irfft(rfft(pred[s,:2], axis=1) * np.exp(-1j*omega*tau[2*s]))
        shifted_pred[s,2] = irfft(rfft(pred[s,2]) * np.exp(-1j*omega*tau[2*s+1]))

    ########################### PLOT
    C0 = 'lightcoral'
    C1 = 'red'
    C2 = 'skyblue'
    C3 = 'Blue'
    ########## INITIALIZE FIGURE
    fig = plt.figure(figsize=(4.5, 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=(1, 3),top=0.93, bottom=0.08, left=0.06, right=0.96, wspace=0.1, hspace=0.1)
    gs0 = gs[0].subgridspec(1, 2)
    gs1 = gs[1].subgridspec(1, 3)

    time = np.arange(nt) * delta
    scale = 0.5 / np.amax(data)
    comps = ['Vertical', 'Radial', 'Tangential']
    ########## PLOT WAVEFORM
    for i_comp in range(nc):
        ax = fig.add_subplot(gs1[i_comp], frameon=False)

        for i_stat in range(ns):
            scale = 0.4 / np.amax(data[i_stat])
            obs_tmp = data[i_stat, i_comp] * scale
            ax.plot(time, obs_tmp + i_stat, c='k', lw=0.5) 
            #
            pred_tmp = shifted_pred[i_stat, i_comp] * scale
            ax.plot(time, pred_tmp + i_stat, c=C1, lw=0.5)

            obs_tmp2_sum = np.sum(obs_tmp**2)
            diff = obs_tmp - pred_tmp
            vr = 1 - np.sum(diff**2) / obs_tmp2_sum
            text_vr = 'WVR=%2.1f ' % (vr*100) 
            if i_comp == 0:
                ax.text( x = 0, y = i_stat - .5, s = '$\\tau$1=%.2f,$\\tau$2=%.2f, k=%.1f' % \
                           (tau[2*i_stat], tau[2*i_stat+1], noise[i_stat]), c='black', fontsize=9)                

        ax.set_yticks(range(ns))    
        ax.tick_params(axis='y', which=u'both',length=0)
        if i_comp == 0:
            ax.set_yticklabels([s.network+'.'+s.station for s in stations])
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0,75,150])
        ax.set_xticklabels([0,75,150])
        ax.set_ylim(-.7, ns-.5)
        ax.plot([0, nt], [-.7, -.7], lw=2, c='k')
        ax.set_xlim(0, max(time+1))
        ax.set_xlabel('Time (s)')
        ax.set_title(comps[i_comp])

        if i_comp == 0: 
            ax.annotate('b)', xy=(0, 1), xycoords='axes fraction', ha='right', fontweight='bold', fontsize=12)
 
    #######MT solution from this study
    mt = MomentTensor.from_values(rtp2ned(mij_sol))
    mt_decom = mt.standard_decomposition()
    c_ISO, c_DC, c_CLVD = 100*mt_decom[0][1], 100*mt_decom[1][1], 100*mt_decom[2][1]
    mw = MT_mtuq(mij_sol).magnitude()

    ax_dev = fig.add_subplot(gs0[0], frame_on=False, aspect='equal', xticks=[], yticks=[])
    beachball.plot_beachball_mpl( mt.m6(), ax_dev, beachball_type='deviatoric',\
        size=8*mw, position=(0.5,.5),\
        alpha=1, edgecolor = C0, color_t=C0, linewidth=1.0)

    beachball.plot_beachball_mpl( mt.m6(), ax_dev, beachball_type='dc',\
        size=8*mw, position=(0.5,.5),\
        alpha=1, edgecolor = 'r', color_t='none', color_p='none', linewidth=1.0)

    ax_dev.set_xlim(0, 2.0)
    ax_dev.set_ylim(0, 1.0)
    ax_dev.text(0.1,0.98,s='The MT solution', fontsize=10)
    ax_dev.annotate('a)', xy=(0, 0.98), xycoords='axes fraction', ha='right', fontweight='bold', fontsize=12)

    ##
    obs_data2_sum = np.sum(data**2)
    diff = data - shifted_pred
    vr = 1 - np.sum(diff**2) / obs_data2_sum

    ax_txt = fig.add_subplot(gs0[1], frame_on=False, aspect='equal', xticks=[], yticks=[])
    ax_txt.text(-.5,0.1,'Depth = %.1f km, Mw = %.2f' % (evdp_in_km, mw), fontsize=10)
    ax_txt.text(-.5,0.25,'Percent DC = %.1f' % c_DC, fontsize=10)
    ax_txt.text(-.5,0.4,'Percent CLVD = %.1f' % c_CLVD, fontsize=10)
    ax_txt.text(-.5,0.55,'Percent ISO = %.1f' % c_ISO, fontsize=10)
    ax_txt.text(-.5,0.7,'VR = %1.1f%%' % (vr*100), fontsize=10)
    plt.savefig(figure_fname, bbox_inches='tight', dpi=300)

def plot_data_covariance_matrix(sigma_in, stations, npts, figname, reference_matrix=None):
    comps = ['Z', 'R', 'T']
    station_list = [s for s in stations]
    ns,nc = sigma_in.shape
    
    sigma_ref = np.fromfile('noise_std_sw_sigma.bin').reshape(8,3)
    sigma_ref = sigma_ref**2*1.0e12    
    
    sigma = sigma_in**2 * 1.0e12 
 
    print(sigma)

    fig,axes = plt.subplots(nc,ns, sharex=True, sharey=True, figsize=(9,2.5), subplot_kw={'xticks': [0,100,200], 'yticks': [0,100,200]})
    norm = matplotlib.colors.Normalize(vmin=np.min(sigma), vmax=np.max(sigma))
    #cm = plt.get_cmap('tab20')
    cm = copy.copy( plt.get_cmap('copper').reversed())
    # cm = plt.get_cmap('jet')
    #cm.set_under('lightgray')
    for ist in range(ns):
       for ic in range(nc):
          if reference_matrix is not None:
              cov_i = reference_matrix[ist,ic] * (sigma[ist,ic] - sigma_ref[ist,ic])
              # cov_i = np.zeros((nt,nt))
              # for j_1 in range(npts):
              #     for j_2 in range(npts):
              #         if abs(j_1-j_2)<2:
              #         if j_1==j_2:
              #            cov_i[j_1,j_2] = sigma[ist,ic]
              # cov_i[cov_i==0.0] =np.nan
              im = axes[ic,ist].imshow(cov_i, vmin=np.min(sigma), vmax=np.max(sigma), cmap =cm)
          else:
              color = cm(norm(sigma[ist,ic]))
              im=axes[ic,ist].plot(np.arange(npts),np.arange(npts), color=color,linewidth=1.)
          if ist == 0 and ic == 1:
              # axes[ic,ist].set_ylabel(comps[ic])
              axes[ic,ist].set_ylabel('Time (s)')
          if ist == int(ns/2):
              axes[ic,ist].set_xlabel('Time (s)')
          axes[0,ist].set_title(station_list[ist],fontsize=9)
          axes[ic,ist].set_xlim([0,npts])
          axes[ic,ist].set_ylim([0,npts])
          plt.gca().invert_yaxis()
           
    axes[0,0].annotate('Z', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
    axes[1,0].annotate('R', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
    axes[2,0].annotate('T', xy=(0.25, 0.75), xycoords='axes fraction', ha='right')
        
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.81, 0.1, 0.02, 0.8])
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=norm)
    cb.set_label(label='Covariance amplitude / $10^{12}$',fontsize=10)
    #plt.colorbar(im, cax=cax, ax = axes[-1,-1])
    plt.savefig(figname, dpi = 300, bbox_inches = 'tight')

