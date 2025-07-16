
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyproj import Geod

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