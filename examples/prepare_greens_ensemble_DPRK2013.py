#!/usr/bin/env python
"""
Prepare Green's function ensemble for ModelErrorCovariance — DPRK 2013 event.

Each station uses its own regional 1D velocity model (Model96 format):
    IC.MDJ  → MDJ.mod
    IU.BJT  → BJT.mod
    IU.HIA  → HIA.mod
    IU.INCN → INCN.mod
    IU.INU  → INU.mod
    IU.MAJO → MAJO.mod
    IC.TJN  → TJN.mod

Workflow
--------
  Step 1 — perturb each station's velocity model (M realisations)
  Step 2 — compute CPS Green's functions for each station's ensemble
  Step 3 — load and process each ensemble into (M, 1, nc, ne, nt) arrays
  Step 4 — build ModelErrorCovariance via from_station_groups

Run from the hibasin_cps/ directory:
    python prepare_greens_ensemble_DPRK2013.py
"""

import os
import glob
import numpy as np

from mtuq import read, open_db
from mtuq.event import Origin
from mtuq.process_data import ProcessData
from mtuq.util.cap import parse_station_codes

from hibasin.util.greens_ensemble import (
    perturb_velocity_model,
    compute_cps_greens,
    load_greens_ensemble,
    plot_velocity_model_ensemble,
    save_model_ensemble_nc4,
)
from hibasin.util.covariance_matrix_Cm import covariance_matrix_Cm
from hibasion.util.math import ned2rtp

# ---------------------------------------------------------------------------
# Paths — adjust if your data lives elsewhere
# ---------------------------------------------------------------------------

HIBASIN_ROOT = '/Users/u7091895/Documents/Research/BayMTI/HiBasin/'

path_data    = os.path.join(HIBASIN_ROOT, 'data/20130212025751000/*.BH[ZRT].sac')
path_weights = os.path.join(HIBASIN_ROOT, 'data/20130212025751000/weights_surf.dat')
CPS_DATABASE = os.path.join(HIBASIN_ROOT, 'data/grn_2013_2d/mdj3/')

# Directory containing the per-station Model96 files
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script

# ---------------------------------------------------------------------------
# Station → velocity model mapping  (STA part of NET.STA → .mod file)
# ---------------------------------------------------------------------------

STATION_MODELS = {
    'MDJ':  os.path.join(MODEL_DIR, 'MDJ.mod'),
    'BJT':  os.path.join(MODEL_DIR, 'BJT.mod'),
    'HIA':  os.path.join(MODEL_DIR, 'HIA.mod'),
    'INCN': os.path.join(MODEL_DIR, 'INCN.mod'),
    'INU':  os.path.join(MODEL_DIR, 'INU.mod'),
    'MAJO': os.path.join(MODEL_DIR, 'MAJO.mod'),
    'TJN':  os.path.join(MODEL_DIR, 'TJN.mod'),
}

# ---------------------------------------------------------------------------
# Ensemble parameters
# ---------------------------------------------------------------------------

NUM_MODELS  = 300   # number of perturbed models per station (M)
PERTURB_PCT = 5     # ±5 % Gaussian perturbation amplitude
DELTA       = 1.0   # target sampling interval (s) — must match data_sw
N_CORES     = 10    # parallel CPS workers

# ---------------------------------------------------------------------------
# Event origin  (identical to EMCEE.FullMomentTensor.DPRK2013.py)
# ---------------------------------------------------------------------------

origin = Origin({
    'time':       '2013-02-12T02:57:51.272500Z',
    'latitude':   41.2921,
    'longitude':  129.0730,
    'depth_in_m': 500.,
})

# ---------------------------------------------------------------------------
# ProcessData  (identical to EMCEE.FullMomentTensor.DPRK2013.py)
# ---------------------------------------------------------------------------

process_sw = ProcessData(
    filter_type   = 'Bandpass',
    freq_min      = 0.02,
    freq_max      = 0.05,
    pick_type     = 'CPS_metadata',
    CPS_database  = CPS_DATABASE,
    window_type   = 'surface_wave',
    window_length = 350,
    capuaf_file   = path_weights,
    apply_scaling = False,
)

# ---------------------------------------------------------------------------
# Load stations (same order as the MCMC run)
# ---------------------------------------------------------------------------

print('Reading data to obtain station list ...')
data = read(path_data, format='sac',
            event_id='20130212025751000',
            station_id_list=parse_station_codes(path_weights),
            tags=['units:m', 'type:displacement'])
data.sort_by_distance()
stations = data.get_stations()

print(f'Stations: {[s.network + "." + s.station for s in stations]}\n')

# ---------------------------------------------------------------------------
# Per-station GF ensemble computation
# ---------------------------------------------------------------------------

greens_ensemble_by_group = {}
group_station_list       = {}

for sta in stations:
    sta_code = sta.station          # e.g. 'MDJ'
    net_sta  = f'{sta.network}.{sta_code}'   # e.g. 'IC.MDJ'

    if sta_code not in STATION_MODELS:
        print(f'Skipping {net_sta}: no velocity model defined in STATION_MODELS')
        continue

    model_file = STATION_MODELS[sta_code]
    work_dir   = os.path.join(MODEL_DIR, 'work', sta_code)

    print(f'--- {net_sta}  ({model_file}) ---')

    # Step 1: perturb velocity model for this station
    model_paths = perturb_velocity_model(
        model_file  = model_file,
        num_models  = NUM_MODELS,
        perturb_pct = PERTURB_PCT,
        output_dir  = os.path.join(work_dir, 'models'),
        seed        = 0,           # fixed seed for reproducibility
    )

    # Plot and save perturbed models for this station
    plot_velocity_model_ensemble(
        model_paths     = model_paths,
        reference_model = model_file,
        figname         = os.path.join(work_dir, f'{sta_code}_models.png'),
    )
    save_model_ensemble_nc4(
        model_paths     = model_paths,
        reference_model = model_file,
        save_path       = os.path.join(work_dir, f'{sta_code}_models.nc4'),
    )

    # Step 2: compute CPS elementary GFs for this single station only.
    # Passing stations=[sta] means the dfile has one distance row and
    # fsel96/f96tosac produces SAC files for this station alone.
    gf_dirs = compute_cps_greens(
        model_paths = model_paths,
        output_dir  = os.path.join(work_dir, 'gf'),
        origin      = origin,
        stations    = [sta],       # single station
        dt          = 0.2,         # CPS sampling interval (must be ≤ DELTA)
        npts        = 4096,        # long enough for the 200-s surface-wave window
        t0          = 0.0,
        vred        = 0,
        n_cores     = N_CORES,
    )

    # Step 3: load, apply process_sw, resample → (M, 1, nc, ne, nt)
    ens_path = os.path.join(work_dir, f'{sta_code}_ensemble.npy')
    ens = load_greens_ensemble(
        gf_db_dirs   = gf_dirs,
        stations     = [sta],
        origin       = origin,
        process_data = process_sw,
        delta        = DELTA,
        save_path    = ens_path,
    )
    # ens shape: (NUM_MODELS, 1, nc, ne, nt)

    print(f'  ensemble shape: {ens.shape}  saved {ens_path}\n')

# # ---------------------------------------------------------------------------
# # Step 4: build ModelErrorCovariance
# # ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# How to reload model_cov without rerunning CPS (e.g. in the MCMC script)
# ---------------------------------------------------------------------------
# Each station's ensemble is saved as a .npy file.  To rebuild model_cov:
#
# print('Building ModelErrorCovariance ...')

greens_ensemble_by_group = {}
group_station_list = {}
for sta in stations:
    net_sta = f'{sta.network}.{sta.station}'
    ens = np.load(f'work/{sta.station}/{sta.station}_ensemble.npy')
    greens_ensemble_by_group[net_sta] = ens
    group_station_list[net_sta] = [net_sta]

model_cov = covariance_matrix_Cm.from_station_groups(
    greens_ensemble_by_group, group_station_list, stations)

# Then in the MCMC setup:
# mij_abs  = best_mt.as_vector() * solver.M00
# cov_inv, log_cov_det = model_cov.calc_InversionDeterminant(
#     cov_d, noise_std, mij_abs)

# ---------------------------------------------------------------------------
# Plot C_m for a moment tensor
# ---------------------------------------------------------------------------

#use the mij for DORK2013 event as the "initial" moment tensor
mij_prior = ned2rtp(np.array([22.74, 30.00, 8.77, -9.13, -3.72, -6.22])) * 1.0e14

print('Plotting C_m for a moment tensor ...')
model_cov.plot_model_covariance_matrix(
    mij     = mij_prior,
    figname = os.path.join(MODEL_DIR, 'Cm_matrix_mt_itr0.png'),
)
print(model_cov.calc_Cm(mij_prior).shape)  # (nd, nd)
print('\nDone.')
