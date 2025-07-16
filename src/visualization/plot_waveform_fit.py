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
from src.util.math import to_lune, Tashiro2MT6,Tashiro2MT6_vec,ned2rtp, rtp2ned2, rtp2ned

def plot_waveform_fit(mij_sol, data, greens, stations, noise, tau, figure_fname, evdp_in_km=33):
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