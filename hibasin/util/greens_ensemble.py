"""
Green's function ensemble builder for earth-model uncertainty (C_m).

Typical workflow
----------------
Step 1 — generate M perturbed 1D Earth models for a given uncertainty level
         (e.g. ±5 % velocity perturbations):

    from hibasin.util.greens_ensemble import perturb_velocity_model
    model_paths = perturb_velocity_model(
        model_file='sw_vel',
        num_models=300,
        perturb_pct=5,
        output_dir='./models',
        seed=0,
    )
    # Writes models/00000.mod, models/00001.mod, ...

Step 2 — compute elementary Green's functions for each perturbed model via CPS:

    from hibasin.util.greens_ensemble import compute_cps_greens
    gf_dirs = compute_cps_greens(
        model_paths = model_paths,
        output_dir  = './zagreb20',
        origin      = origin,      # mtuq Origin — supplies depth and coordinates
        stations    = stations,    # mtuq station list — supplies distances
        dt          = 0.5,         # CPS sampling interval (s)
        npts        = 1024,        # CPS time samples
        vred        = 5.7,         # velocity reduction (km/s)
        n_cores     = 10,
    )
    # Produces the directory layout MTUQ's open_db(format='CPS') expects:
    #   ./zagreb20/00000/0005/*.sac   (model 00000, depth 0.5 km → '0005')
    #   ./zagreb20/00001/0005/*.sac   ...
    # Depth dir = int(depth_km * 10) zero-padded to 4 digits.

Step 3 — assemble the ensemble numpy array:

    from hibasin.util.greens_ensemble import load_greens_ensemble
    greens_ensemble = load_greens_ensemble(
        gf_db_dirs   = gf_dirs,
        stations     = stations,
        origin       = origin,
        process_data = process_sw,   # same ProcessData as observed data
        delta        = 1.0,          # resample to 1 Hz
        save_path    = 'greens_ensemble.npy',
    )
    # greens_ensemble : (M, ns, nc, ne, nt)
    # ne = 6 for full moment tensor (Mxx, Myy, Mzz, Mxy, Mxz, Myz)

Step 4 — build ModelErrorCovariance:

    from hibasin.util.covariance_matrix_Cm import ModelErrorCovariance
    model_cov = ModelErrorCovariance.from_same_model(greens_ensemble, stations)
"""

import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
from mtuq import open_db
from mtuq.misfit.waveform import level2


# ---------------------------------------------------------------------------
# Step 1 — model perturbation
# ---------------------------------------------------------------------------

def perturb_velocity_model(model_file, num_models, perturb_pct, output_dir, seed=0):
    """
    Generate an ensemble of perturbed 1D velocity models in CPS Model96 format.

    Thickness, bulk modulus (kappa), and shear modulus (mu) are perturbed
    independently per layer with Gaussian noise scaled to perturb_pct % of
    the reference value.  Number of layers, density and Q-factors are left unchanged, following
    Phạm and Tkalčić (2021).

    Parameters
    ----------
    model_file : str
        Reference velocity model.  Two formats are accepted:

        CPS Model96 or
        Simple whitespace-delimited (no header):
            thickness(km)  vp(km/s)  vs(km/s)  rho(g/cm^3)  qka  qmu
    num_models : int
        Number of perturbed models M to generate.
    perturb_pct : float
        Perturbation amplitude as a percentage of the reference value
        (e.g. 5 for ±5 % 1-sigma Gaussian perturbations).
    output_dir : str
        Directory to write the .mod files.  Created if it does not exist.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    model_paths : list of str
        Sorted paths to the written Model96 files (00000.mod, 00001.mod, ...).
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    thick, vp, vs, rho, qka, qmu = _read_velocity_model(model_file)
    mu    = rho * vs ** 2
    kappa = rho * vp ** 2 - 4.0 * mu / 3.0

    ref = np.array([thick, kappa, mu, rho, qka, qmu])   # (6, nlayers)
    std = (perturb_pct / 100.0) * ref

    model_paths = []
    for m in range(num_models):
        dev = np.random.normal(0.0, 1.0, ref.shape)
        dev[3:6, :] = 0.0          # density and Q unchanged
        perturbed = ref + dev * std
        path = os.path.join(output_dir, f'{m:05d}.mod')
        _write_model96(perturbed, path)
        model_paths.append(path)

    print(f"Wrote {num_models} perturbed models to '{output_dir}/'")
    return sorted(model_paths)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_velocity_model(model_file):
    """
    Read a velocity model file in either CPS Model96 or simple column format.

    CPS Model96 is detected by the 'MODEL' prefix on the first line.  It has
    a 12-line header (11 text lines + 1 column-label line) followed by data
    rows:  H(KM)  VP(KM/S)  VS(KM/S)  RHO(GM/CC)  QP  QS  ...

    Simple format has no header:  thickness  vp  vs  rho  qka  qmu

    Returns
    -------
    thick, vp, vs, rho, qka, qmu : ndarray (nlayers,)
    """
    with open(model_file) as fh:
        first = fh.readline().strip()

    # Model96: 11-line text header + 1 column-label line = 12 lines to skip
    skiprows = 12 if first.upper().startswith('MODEL') else 0

    data = np.loadtxt(model_file, skiprows=skiprows)
    thick = data[:, 0]
    vp    = data[:, 1]
    vs    = data[:, 2]
    rho   = data[:, 3]
    qka   = data[:, 4]
    qmu   = data[:, 5]
    return thick, vp, vs, rho, qka, qmu


def _write_model96(vel_model, fname):
    """Write a 1D model array (6, nlayers) to CPS Model96 format."""
    thick = vel_model[0]
    kappa = vel_model[1]
    mu    = vel_model[2]
    rho   = vel_model[3]
    vp    = np.sqrt((kappa + 4.0 * mu / 3.0) / rho)
    vs    = np.sqrt(mu / rho)
    qka   = vel_model[4]
    qmu   = vel_model[5]

    with open(fname, 'w') as fid:
        fid.write(
            'MODEL.01\n'
            'Perturbed 1D velocity model\n'
            'ISOTROPIC\n'
            'KGS\n'
            'FLAT EARTH\n'
            '1-D\n'
            'CONSTANT VELOCITY\n'
            'LINE08\nLINE09\nLINE10\nLINE11\n'
        )
        fid.write(
            'H(KM)    VP(KM/S) VS(KM/S) RHO(GM/CC)  QP       QS    '
            'ETAP     ETAS   FREFP    FREFS\n'
        )
        for n in range(len(thick)):
            fid.write(
                '%-8.2f %-8.3f %-8.3f %-8.3f %-8.1f %-8.1f '
                '%-8.1f %-8.1f %-8.1f %-8.1f\n' % (
                    thick[n], vp[n], vs[n], rho[n],
                    qka[n], qmu[n], 0.0, 0.0, 1.0, 1.0,
                )
            )

            
# ---------------------------------------------------------------------------
# Step 2 — CPS Green's function computation (replaces genCPSElemGFs.sh)
# ---------------------------------------------------------------------------

def compute_cps_greens(
    model_paths,
    output_dir,
    origin,
    stations,
    dt=0.5,
    npts=1024,
    t0=0.0,
    vred=5.7,
    n_cores=10,
):
    """
    Compute elementary Green's functions for an ensemble of perturbed 1D Earth
    models using CPS tools.

    Parameters
    ----------
    model_paths : list of str
        Paths to CPS Model96 (.mod) files — typically the output of
        perturb_velocity_model().
    output_dir : str
        Root directory where per-model .GF sub-directories are created.
        E.g. output_dir='mdj2' → mdj2/00000.GF/, mdj2/00001.GF/, ...
    origin : mtuq Origin
    stations : list
        MTUQ station list in the same order as the observed data. 
    dt : float
        CPS sampling interval in seconds (dfile column 2).  Use a value
        smaller than or equal to the target delta of the observed data so
        that resampling in load_greens_ensemble() can work correctly.
    npts : int
        Number of time samples per elementary seismogram (dfile column 3).
        Must be long enough to capture the full surface-wave window.
    t0 : float
        Start-time offset in seconds before the first sample (dfile column 4).
    vred : float
        Velocity reduction in km/s (dfile column 5).  Set to 0 to disable.
    n_cores : int
        Number of parallel workers.  Each worker handles one perturbed model.

    Returns
    -------
    gf_dirs : list of str
        Sorted paths to the per-model .GF directories containing SAC files
        produced by f96tosac -G, ready for load_greens_ensemble().
    """
    os.makedirs(output_dir, exist_ok=True)

    # Write the shared dfile: one row per station
    dfile_path = _write_dfile(output_dir, origin, stations, dt, npts, t0, vred)

    evdp_km = origin.depth_in_m / 1000.0
    station_codes = [f"{s.network}.{s.station}" for s in stations]

    # Build per-model argument tuples.
    # Directory layout: output_dir/{model_name}/{depth_dir}/*.sac
    # e.g.  mdj2/00000/0005/*.sac  (0005 = 0.5 km × 10, zero-padded to 4 digits)
    args_list  = []
    model_dirs = []
    for mod_path in model_paths:
        mod_name  = os.path.splitext(os.path.basename(mod_path))[0]  # '00000'
        model_dir = os.path.abspath(os.path.join(output_dir, mod_name))
        os.makedirs(model_dir, exist_ok=True)
        args_list.append((
            os.path.abspath(mod_path),
            model_dir,
            os.path.abspath(dfile_path),
            station_codes,
            evdp_km,
        ))
        model_dirs.append(model_dir)

    print(f"Running CPS for {len(model_paths)} models on {n_cores} cores ...")
    completed = 0
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        futures = {
            executor.submit(_run_cps_one_model, *args): model_dir
            for args, model_dir in zip(args_list, model_dirs)
        }
        for future in as_completed(futures):
            future.result()   # re-raises any exception from the worker
            completed += 1
            if completed % 10 == 0 or completed == len(model_paths):
                print(f"  CPS done: {completed}/{len(model_paths)}", flush=True)

    return sorted(model_dirs)


def _write_dfile(output_dir, origin, stations, dt, npts, t0, vred):
    """Write the CPS dfile used by hprep96 (one row per station)."""
    dfile_path = os.path.join(output_dir, 'dfile')
    with open(dfile_path, 'w') as fid:
        for sta in stations:
            dist_m, _, _ = gps2dist_azimuth(
                origin.latitude,  origin.longitude,
                sta.latitude,     sta.longitude,
            )
            dist_km = dist_m / 1000.0
            fid.write(f'{dist_km:.1f} {dt:.2f} {int(npts)} {t0:.1f} {vred:.1f}\n')
    return dfile_path


def _run_cps_one_model(mod_path, model_dir, dfile_path, station_codes, evdp_km):
    """
    Worker: run the full CPS pipeline for one perturbed model.

    Produces the directory structure expected by MTUQ's open_db(format='CPS'):

        model_dir/
          {depth_dir}/        e.g. 0005  (= int(depth_km * 10), 4-digit zero-padded)
            *.sac             SAC files from f96tosac -G

    Pipeline: hprep96 → hspec96 → hpulse96 → fsel96 → f96tosac -G
    """
    mod_name = os.path.basename(mod_path)

    # Depth subdirectory: MTUQ expects model_dir/{depth}/  where depth is
    # depth_km * 10 formatted as a 4-digit zero-padded integer.
    # e.g. 0.5 km → '0005',  5.0 km → '0050',  10.0 km → '0100'
    depth_str = f'{int(round(evdp_km * 10)):04d}'
    depth_dir = os.path.join(model_dir, depth_str)
    os.makedirs(depth_dir, exist_ok=True)

    # Symlink model file and dfile into the depth directory so CPS can find them
    for target, link in [
        (mod_path,   os.path.join(depth_dir, mod_name)),
        (dfile_path, os.path.join(depth_dir, 'dfile')),
    ]:
        if not os.path.exists(link):
            os.symlink(target, link)

    def _run(cmd, stdout=None):
        result = subprocess.run(
            cmd, cwd=depth_dir,
            stdout=stdout,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CPS failed in {depth_dir}\n"
                f"  cmd : {' '.join(str(c) for c in cmd)}\n"
                f"  stderr: {result.stderr.decode(errors='replace')}"
            )
        return result

    # hprep96: prepare spectral input files
    _run(['hprep96', '-M', mod_name, '-d', 'dfile',
          '-HS', f'{evdp_km:.4f}', '-HR', '0.0', '-EQEX', '-R'])

    # hspec96: compute wavenumber integration spectra
    hspec_out = os.path.join(depth_dir, 'hspec96.out')
    with open(hspec_out, 'wb') as fout:
        _run(['hspec96'], stdout=fout)

    # hpulse96: convert spectra to time-domain displacement seismograms
    # -D = displacement output, -i = impulse source
    hpulse_out = os.path.join(depth_dir, 'hpulse96.out')
    with open(hpulse_out, 'wb') as fout:
        _run(['hpulse96', '-D', '-i'], stdout=fout)

    # fsel96 → f96tosac -G: extract per-station GFs and convert to SAC.
    # SAC files are written to depth_dir (cwd), named by the f96 header
    # (e.g. NN.SSSS.ZSS.sac, NN.SSSS.ZDS.sac, ...).
    with open(hpulse_out, 'rb') as fh:
        pulse_bytes = fh.read()

    for i, code in enumerate(station_codes, start=1):
        f96_result = subprocess.run(
            ['fsel96', '-NS', str(i)],
            input=pulse_bytes,
            cwd=depth_dir,
            capture_output=True,
        )
        if f96_result.returncode != 0:
            raise RuntimeError(
                f"fsel96 failed for station {code} in {depth_dir}:\n"
                f"  {f96_result.stderr.decode(errors='replace')}"
            )

        sac_result = subprocess.run(
            ['f96tosac', '-G'],
            input=f96_result.stdout,
            cwd=depth_dir,
            capture_output=True,
        )
        if sac_result.returncode != 0:
            raise RuntimeError(
                f"f96tosac failed for station {code} in {depth_dir}:\n"
                f"  {sac_result.stderr.decode(errors='replace')}"
            )

    # Remove intermediate CPS files from depth_dir; keep SAC files
    for path in glob.glob(os.path.join(depth_dir, '*96.out')):
        os.remove(path)
    for path in glob.glob(os.path.join(depth_dir, '*96.???')):
        os.remove(path)


# ---------------------------------------------------------------------------
# Step 3 — GF ensemble loader
# ---------------------------------------------------------------------------

def load_greens_ensemble(
    gf_db_dirs,
    stations,
    origin,
    process_data,
    components=None,
    delta=None,
    save_path=None,
):
    """
    Load M CPS Green's function databases, apply processing, and stack into
    the ensemble array required by ModelErrorCovariance.

    Parameters
    ----------
    gf_db_dirs : list of str
        Paths to M CPS Green's function database directories, one per
        perturbed Earth model.  Each directory must contain the SAC files
        produced by compute_cps_greens().
        Example: sorted(glob.glob('mdj2/0005/?????.GF'))
    stations : list
        MTUQ station list in the same order as the observed data.
    origin : mtuq Origin
        Event origin (time, latitude, longitude, depth_in_m).
    process_data : mtuq ProcessData
        The same ProcessData object used to process the observed data.
        Applied identically to every ensemble member.
    components : list of str or None
        Seismic components to extract (e.g. ['Z', 'R', 'T']).  If None,
        derived automatically from the first loaded model.
    delta : float or None
        Target sampling interval in seconds (e.g. 1.0 for 1 Hz).  Must
        match the sampling interval of the processed observed data.
        If None, no resampling is applied.
    save_path : str or None
        If given, saves the result with np.save ('.npy' appended if absent).

    Returns
    -------
    greens_ensemble : ndarray (M, ns, nc, ne, nt)
        Processed GF ensemble ready for ModelErrorCovariance.
    """
    M = len(gf_db_dirs)
    if M == 0:
        raise ValueError("gf_db_dirs is empty — no GF databases found.")

    greens_ensemble = None

    for m, db_path in enumerate(gf_db_dirs):
        model_name = os.path.basename(db_path.rstrip(os.sep))
        db = open_db(db_path, format='CPS', model=model_name)
        greens = db.get_greens_tensors(stations, origin)
        greens_proc = greens.map(process_data)

        if delta is not None:
            sampling_rate = 1.0 / delta
            for s in range(len(stations)):
                greens_proc[s].resample(sampling_rate)
                greens_proc[s].station._refresh('delta', delta)

        if components is None:
            components = level2._get_components(greens_proc)

        arr = level2._get_greens(greens_proc, stations, components)  # (ns, nc, ne, nt)

        if greens_ensemble is None:
            ns, nc, ne, nt = arr.shape
            greens_ensemble = np.zeros((M, ns, nc, ne, nt), dtype=np.float64)
            print(f"GF ensemble shape: ({M}, {ns}, {nc}, {ne}, {nt})")

        greens_ensemble[m] = arr

        if (m + 1) % 10 == 0 or m == M - 1:
            print(f"  Loaded {m + 1}/{M}", flush=True)

    if save_path is not None:
        if not save_path.endswith('.npy'):
            save_path += '.npy'
        np.save(save_path, greens_ensemble)
        print(f"Saved → {save_path}  shape={greens_ensemble.shape}")

    return greens_ensemble



