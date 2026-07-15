import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["savefig.dpi"] = 300
rcParams["font.size"]   = 10

from mtuq import MomentTensor
from mtuq.graphics import _plot_beachball_matplotlib

plt.rcParams['savefig.facecolor'] = 'white'


def plot_misfit_depth(depths_km, mean_log_probs, mean_mts, figure_fname,
                      event_id=''):
    """
    Plot mean log-probability vs depth with a beachball of the mean moment
    tensor placed at each depth point on the curve.

    The best-fit depth (highest mean log_prob) is highlighted with a red
    beachball and a dashed vertical line.  All other depths are shown in
    light gray.

    Parameters
    ----------
    depths_km : array-like of float
        Depth values in km, one per MCMC run.
    mean_log_probs : array-like of float
        Mean log-probability from the posterior half of each MCMC chain.
    mean_mts : list of array-like
        Mean moment tensor at each depth as a 6-element vector in rtp
        convention [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp], as returned by
        ``mtuq_MomentTensor.as_vector()``.
    figure_fname : str
        Output figure path / filename.
    event_id : str, optional
        Event identifier used in the figure title.
    """
    depths_km      = np.asarray(depths_km,      dtype=float)
    mean_log_probs = np.asarray(mean_log_probs,  dtype=float) / 1.0e4
    ndepth         = len(depths_km)

    if ndepth != len(mean_log_probs) or ndepth != len(mean_mts):
        raise ValueError(
            'depths_km, mean_log_probs and mean_mts must have the same length.')

    best_idx      = int(np.argmax(mean_log_probs))
    best_depth_km = depths_km[best_idx]

    # ── Figure and main axes ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 8))

    ax.plot(depths_km, mean_log_probs,
            color='k', linewidth=1.5, zorder=2)
    ax.scatter(depths_km, mean_log_probs,
               color='k', s=18, zorder=3)
    ax.axvline(best_depth_km,
               color='r', linestyle='--', linewidth=1.5,
               label='Best depth: %d km' % int(best_depth_km), zorder=2)

    ax.set_xlabel('Depth (km)', fontsize=13)
    ax.set_ylabel('Mean log-probability / $10^4$', fontsize=13)
    title = ('%s  —  MCMC mean log-probability vs depth' % event_id
             if event_id else 'MCMC mean log-probability vs depth')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Extra vertical margin so beachballs centred on the curve don't overflow
    lp_range = mean_log_probs.max() - mean_log_probs.min() or 1.0
    ax.set_ylim(mean_log_probs.min() - 0.15 * lp_range,
                mean_log_probs.max() + 0.15 * lp_range)
    # Horizontal padding so edge beachballs are not clipped
    ax.set_xlim(depths_km[0] - 0.6, depths_km[-1] + 0.6)

    plt.tight_layout()
    fig.canvas.draw()   # lock layout before computing transforms

    # ── Beachball size (figure-fraction units) ────────────────────────────────
    unique_spacings = np.diff(np.sort(np.unique(depths_km)))
    depth_spacing   = (float(unique_spacings.min()) if len(unique_spacings) else depths_km[0])

    x_data_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax_frac_w    = ax.get_position().width          # axes width in fig fraction
    bb_size      = 0.75 * depth_spacing / (x_data_range / ax_frac_w)
    bb_size      = min(bb_size, 0.22)               # cap to avoid overflow

    # ── Draw one beachball per depth ─────────────────────────────────────────
    for i, (d, lp, mij) in enumerate(zip(depths_km, mean_log_probs, mean_mts)):
        mij = np.asarray(mij, dtype=float)

        # Data point → display pixels → figure fraction
        xy_disp = ax.transData.transform((d, lp))
        xy_fig  = fig.transFigure.inverted().transform(xy_disp)

        # Inset axes centred on the data point
        axbb = fig.add_axes([xy_fig[0] - bb_size / 2,
                             xy_fig[1] - bb_size / 2,
                             bb_size, bb_size])
        axbb.set_xlim(-0.7, 0.7)
        axbb.set_ylim(-0.7, 0.7)
        axbb.set_aspect('equal')
        axbb.axis('off')

        is_best = (i == best_idx)
        color   = 'red' if is_best else 'black'

        mt = MomentTensor(mij, convention='USE')
        _plot_beachball_matplotlib(filename=None, mt_arrays=mt,
                                   fig=fig, ax=axbb, color=color, scale=4.8)

    plt.savefig(figure_fname, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved depth misfit figure: %s' % figure_fname)
