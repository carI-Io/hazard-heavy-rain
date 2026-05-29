#!/usr/bin/env python3
"""
more_chicago_hyetograph.py
==========================
MORE Dataset: Chicago Hyetograph Maps — 1h & 3h Event Durations
MOloch-downscaled ERA5 REanalysis (1991–2020)

For each grid point across Italy, reads the GEV-fitted precipitation depth from
the IDF results produced by more_idf_italy.py, then applies the Chicago
(Keifer & Chu) hyetograph to distribute that depth in time.

Outputs
-------
- Spatial maps of peak intensity  [mm/hr]  for 1h and 3h design storms
- Spatial maps of total depth     [mm]     (sanity check vs. IDF raster)
- Single-point example hyetograph plots
- NetCDF + PNG files ready for GIS / hydraulic modelling
- IDF data inspection report

Inputs required (produced by more_idf_italy.py)
-----------------------------------------------
    IDF_results/idf_1h.nc   – return_value(return_period, lat, lon)
    IDF_results/idf_3h.nc
    IDF_results/idf_6h.nc
    IDF_results/idf_12h.nc
    IDF_results/idf_24h.nc

Usage
-----
    python more_chicago_hyetograph.py [--output-dir PATH] [--target-rp 100]
                                      [--skip-plots] [--log-file PATH]
"""

# ── standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm                              # noqa: F401  (used by sub-routines)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS  (override via CLI or environment variables)
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR  = os.environ.get(
    'MORE_OUTPUT_DIR',
    os.path.join('/home/admin_climatecharted_com/data/MOloch', 'IDF_results')
)
LOG_FILE    = os.environ.get('MORE_LOG_FILE', None)

TARGET_RP   = 100                       # return period [years]
DURATIONS   = [1, 3]                    # event durations to process [hours]
ALL_DURATIONS = [1, 3, 6, 12, 24]      # must match available idf_<d>h.nc files

# ── Chicago hyetograph parameters ─────────────────────────────────────────────
CHICAGO_R  = 0.35   # peak-position ratio (r=0.35 common for Italian storms)
CHICAGO_A  = 0.65   # IDF scale factor [mm/hr]
CHICAGO_B  = 0.10   # IDF time offset  [hr]
CHICAGO_N  = 0.72   # IDF decay exponent [–]
CHICAGO_DT = 5.0    # hyetograph timestep [minutes]

# ── IDF parameter fitting ──────────────────────────────────────────────────────
ALL_RETURN_PERIODS  = np.array([5, 10, 25, 50, 100])
B_FIX               = 0.1      # fixed time offset [hr] (Italian LSPP standard)
SAVE_FIT_QUALITY    = True
OVERWRITE_PARAMS    = False
SAVE_CUBE           = False     # export full hyetograph cube to NetCDF (large!)

# ── Physical plausibility caps [mm] ───────────────────────────────────────────
MAX_PHYSICAL_DEPTH = {1: 300, 3: 500, 6: 700, 12: 900, 24: 1200}

# ── Reference point for single-pixel plots ────────────────────────────────────
POINT_NAME = 'Milan'
POINT_LAT  = 45.46
POINT_LON  = 9.19


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_logging(log_file=None):
    fmt     = '%(asctime)s  %(levelname)-8s  %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — IDF DATA INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_idf_data(chicago_dir):
    """
    Inspect annual-max stacks, IDF return-value files, GEV outlier ratios,
    and any existing Chicago outputs.  Writes a plain-text report to
    <OUTPUT_DIR>/inspect_idf_report.txt and logs a summary.
    """
    log.info('── Section 0: IDF data inspection ─────────────────────────────')

    rps = ALL_RETURN_PERIODS.tolist()
    lines = []

    def _log(msg=''):
        log.info('%s', msg)
        lines.append(str(msg))

    _log('=' * 72)
    _log('IDF DATA INSPECTION REPORT')
    _log('=' * 72)

    # ── 1. Annual-maximum files ───────────────────────────────────────────────
    _log()
    _log('── 1. Annual-maximum files (annmax_<d>h.nc) ──────────────────────')
    for dur in ALL_DURATIONS:
        path = os.path.join(OUTPUT_DIR, f'annmax_{dur}h.nc')
        if not os.path.exists(path):
            _log(f'  {dur:>2d}h  MISSING')
            continue
        ds  = xr.open_dataset(path)
        var = f'annmax_tp_{dur}h'
        if var not in ds:
            var = list(ds.data_vars)[0]
        arr = ds[var].values.astype(np.float64)
        n_years, n_lat, n_lon = arr.shape
        _log(
            f'  {dur:>2d}h  shape={arr.shape}  years={n_years}  '
            f'min={np.nanmin(arr):.1f}  p50={np.nanpercentile(arr, 50):.1f}  '
            f'p99={np.nanpercentile(arr, 99):.1f}  max={np.nanmax(arr):.1f} mm  '
            f'NaN={int(np.sum(~np.isfinite(arr)))}  '
            f'zero={int(np.sum(arr == 0))}'
        )
        ds.close()

    # ── 2. IDF return-value files ─────────────────────────────────────────────
    _log()
    _log('── 2. IDF return-value files (idf_<d>h.nc) ───────────────────────')
    _log(
        f'  {"Dur":>4s}  {"RP":>5s}  {"min":>8s}  {"p50":>8s}  {"p95":>8s}  '
        f'{"p99":>8s}  {"max":>14s}  {"NaN":>7s}  {"<=0":>7s}  {">cap":>7s}  {"cap":>5s}'
    )
    _log('  ' + '-' * 90)

    all_caps_ok = True
    for dur in ALL_DURATIONS:
        path = os.path.join(OUTPUT_DIR, f'idf_{dur}h.nc')
        if not os.path.exists(path):
            _log(f'  {dur:>2d}h  MISSING')
            continue
        ds  = xr.open_dataset(path)
        cap = MAX_PHYSICAL_DEPTH[dur]
        for rp in rps:
            if rp not in ds['return_period'].values:
                continue
            arr     = ds['return_value'].sel(return_period=rp).values.astype(np.float64)
            finite  = arr[np.isfinite(arr)]
            n_nan   = int(np.sum(~np.isfinite(arr)))
            n_zero  = int(np.sum(finite <= 0))
            n_huge  = int(np.sum(finite > cap))
            n_land  = int(finite.size)
            pct_bad = (n_zero + n_huge) / n_land * 100 if n_land > 0 else 0.0
            flag    = '  ← OUTLIERS' if (n_huge > 0 or pct_bad > 1.0) else ''
            if n_huge > 0:
                all_caps_ok = False
            _log(
                f'  {dur:>3d}h  RP{rp:>3d}  '
                f'{np.nanmin(arr):>8.1f}  '
                f'{np.nanpercentile(arr, 50):>8.1f}  '
                f'{np.nanpercentile(arr, 95):>8.1f}  '
                f'{np.nanpercentile(arr, 99):>8.1f}  '
                f'{np.nanmax(arr):>14.1f}  '
                f'{n_nan:>7d}  {n_zero:>7d}  {n_huge:>7d}  {cap:>5d}{flag}'
            )
        ds.close()

    _log()
    if all_caps_ok:
        _log('  ✓ No pixels exceed physical caps — depth rasters look clean.')
    else:
        _log('  ✗ Unphysical pixels detected (see ">cap" column above).')
        _log('    The masking patch in Section 4 is REQUIRED.')

    # ── 3. GEV shape parameter diagnosis ─────────────────────────────────────
    _log()
    _log('── 3. GEV outlier fingerprint (ratio RP100/RP10 per duration) ────')
    _log('   Healthy ratio ≈ 1.5–2.5.  Ratio >> 3 → potential blow-up.')
    _log()
    _log(
        f'  {"Dur":>4s}  {"p50(ratio)":>12s}  {"p95(ratio)":>12s}  '
        f'{"p99(ratio)":>12s}  {"max(ratio)":>12s}  {"pixels>5":>10s}'
    )
    _log('  ' + '-' * 68)

    for dur in ALL_DURATIONS:
        path = os.path.join(OUTPUT_DIR, f'idf_{dur}h.nc')
        if not os.path.exists(path):
            continue
        ds    = xr.open_dataset(path)
        rv100 = ds['return_value'].sel(return_period=100).values.astype(np.float64)
        rv10  = ds['return_value'].sel(return_period=10).values.astype(np.float64)
        ds.close()
        ratio = rv100 / np.where(rv10 > 0, rv10, np.nan)
        ratio = ratio[np.isfinite(ratio)]
        _log(
            f'  {dur:>3d}h  '
            f'{np.nanpercentile(ratio, 50):>12.2f}  '
            f'{np.nanpercentile(ratio, 95):>12.2f}  '
            f'{np.nanpercentile(ratio, 99):>12.2f}  '
            f'{np.nanmax(ratio):>12.2f}  '
            f'{int(np.sum(ratio > 5)):>10d}'
        )

    # ── 4. Existing Chicago outputs ───────────────────────────────────────────
    _log()
    _log('── 4. Chicago output files ───────────────────────────────────────')
    if not os.path.isdir(chicago_dir):
        _log('  chicago/ subfolder not found.')
    else:
        for dur in [1, 3]:
            path = os.path.join(chicago_dir, f'chicago_{dur}h_RP100.nc')
            if not os.path.exists(path):
                _log(f'  chicago_{dur}h_RP100.nc  MISSING')
                continue
            ds = xr.open_dataset(path)
            pi = ds['peak_intensity'].values.astype(np.float64)
            td = ds['total_depth'].values.astype(np.float64)
            n_bad = int(np.sum(pi[np.isfinite(pi)] > 1000))
            _log(f'  chicago_{dur}h_RP100.nc')
            _log(
                f'    peak_intensity: min={np.nanmin(pi):.1f}  '
                f'p50={np.nanpercentile(pi, 50):.1f}  '
                f'p99={np.nanpercentile(pi, 99):.1f}  max={np.nanmax(pi):.1f} mm/hr  '
                f'pixels>1000mm/hr={n_bad}'
            )
            _log(
                f'    total_depth:    min={np.nanmin(td):.1f}  '
                f'p50={np.nanpercentile(td, 50):.1f}  '
                f'p99={np.nanpercentile(td, 99):.1f}  max={np.nanmax(td):.1f} mm'
            )
            _log(
                f'    {"✗ peak_intensity has " + str(n_bad) + " pixels > 1000 mm/hr — rerun after patch." if n_bad > 0 else "✓ peak_intensity looks physically plausible."}'
            )
            ds.close()

    # ── 5. Summary ────────────────────────────────────────────────────────────
    _log()
    _log('=' * 72)
    _log('SUMMARY & ACTION ITEMS')
    _log('=' * 72)
    _log(
        '\n1. If ">cap" > 0 for any row in Section 2:\n'
        '     → The MAX_PHYSICAL_DEPTH masking patch in Section 4 is required.\n'
        '     → Re-run this script (Sections 4–7) to regenerate clean outputs.\n\n'
        '2. If max(ratio) >> 5 for short durations (1h, 3h) in Section 3:\n'
        '     → GEV fits for those pixels have ξ >> 0 (Fréchet tail).\n'
        '     → Consider adding a GEV shape-parameter cap (ξ ≤ 0.5) in the\n'
        '       IDF script\'s fit_and_compute_rp(), or raise MAX_PHYSICAL_DEPTH.\n\n'
        '3. If chicago/ outputs show pixels > 1000 mm/hr (Section 4):\n'
        '     → Those were produced BEFORE the patch.  Delete and regenerate.\n\n'
        '4. If annmax files show NaN > 0 (Section 1):\n'
        '     → Check the de-accumulation / masking step in the IDF script.'
    )

    # ── Write report file ─────────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, 'inspect_idf_report.txt')
    with open(report_path, 'w') as fh:
        fh.write('\n'.join(lines))
    log.info('Inspection report written → %s', report_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CHICAGO HYETOGRAPH HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_chicago_hyetograph(
    total_precip_mm: float,
    duration_hours: float,
    dt_minutes: float,
    r: float,
    a: float,
    b: float,
    n: float,
) -> pd.DataFrame:
    """
    Generate a Chicago (Keifer & Chu, 1957) design hyetograph.

    Works from the peak outward on each limb using the IDF cumulative curve
    P(ta) = a * ta / (ta + b)^n [mm].  Incremental depths are slotted back
    around the peak so the largest increments are nearest the peak.

    Returns a DataFrame with columns: time_min, intensity_mm_hr, depth_mm.
    """
    dt_hr  = dt_minutes / 60.0
    n_pre  = int(round(r * duration_hours / dt_hr))
    n_post = int(round((1 - r) * duration_hours / dt_hr))

    def P(ta):
        return a * ta / (ta + b) ** n

    ta_pre      = np.arange(1, n_pre + 1) * dt_hr
    dP_pre      = np.diff(P(ta_pre), prepend=0.0)
    pre_int     = dP_pre[::-1] / dt_hr

    ta_post     = np.arange(1, n_post + 1) * dt_hr
    dP_post     = np.diff(P(ta_post), prepend=0.0)
    post_int    = dP_post / dt_hr

    intensities = np.concatenate([pre_int, post_int])

    computed_depth = np.sum(intensities) * dt_hr
    if computed_depth > 0:
        intensities *= total_precip_mm / computed_depth

    times_min = np.arange(n_pre + n_post) * dt_minutes

    return pd.DataFrame({
        'time_min':        times_min,
        'intensity_mm_hr': intensities,
        'depth_mm':        intensities * dt_hr,
    })


def chicago_peak_intensity(total_precip_mm, duration_hours, dt_minutes,
                            r, a, b, n):
    """Return only the peak intensity [mm/hr] of the Chicago hyetograph."""
    df = create_chicago_hyetograph(
        total_precip_mm, duration_hours, dt_minutes, r, a, b, n
    )
    return float(df['intensity_mm_hr'].max())


def chicago_peak_map(depth_2d, duration_hours, dt_minutes, r, a, b, n):
    """
    Compute Chicago peak intensity for every pixel in a 2-D depth array.

    Parameters
    ----------
    depth_2d : np.ndarray (lat, lon) [mm]

    Returns
    -------
    peak_map   : np.ndarray (lat, lon)    [mm/hr]
    hyeto_cube : np.ndarray (T, lat, lon) [mm/hr]
    time_min   : np.ndarray (T,)          [min]
    """
    template = create_chicago_hyetograph(
        total_precip_mm=1.0,
        duration_hours=duration_hours,
        dt_minutes=dt_minutes,
        r=r, a=a, b=b, n=n
    )
    tmpl = template['intensity_mm_hr'].values

    if not np.any(np.isfinite(tmpl)) or np.nanmax(tmpl) == 0:
        raise ValueError(
            f'chicago_peak_map: degenerate template (all NaN/zero). '
            f'Check IDF parameters: a={a}, b={b}, n={n}'
        )

    hyeto_cube = tmpl[:, None, None] * depth_2d[None, :, :]
    peak_map   = hyeto_cube.max(axis=0)

    return peak_map.astype(np.float32), hyeto_cube.astype(np.float32), template['time_min'].values


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD IDF RASTERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_idf_data(target_rp):
    """
    Load the IDF return-value raster for each duration in DURATIONS.

    Returns
    -------
    idf_data : dict {dur_h: xr.DataArray (lat, lon)}
    """
    log.info('── Section 3: Loading IDF rasters (RP%d) ──────────────────────', target_rp)
    idf_data = {}
    for dur in DURATIONS:
        idf_path = os.path.join(OUTPUT_DIR, f'idf_{dur}h.nc')
        if not os.path.exists(idf_path):
            raise FileNotFoundError(
                f'{idf_path} not found.\n'
                'Run more_idf_italy.py first to produce the IDF rasters.'
            )
        ds = xr.open_dataset(idf_path)
        da = ds['return_value'].sel(return_period=target_rp)
        idf_data[dur] = da.load()
        ds.close()
        log.info(
            '  %dh  RP%d: min=%.1f  max=%.1f  mean=%.1f mm',
            dur, target_rp,
            float(da.min()), float(da.max()), float(da.mean())
        )
    return idf_data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3b — PIXEL-WISE IDF PARAMETER FITTING
# ═══════════════════════════════════════════════════════════════════════════════

def fit_idf_parameters():
    """
    Fit i = a / (t + b)^n pixel-by-pixel via log-linear OLS across all five
    durations and all return periods.

    Saves idf_param_a.nc, idf_param_n.nc and optionally idf_param_fit_quality.nc.

    Returns
    -------
    A_MAP : np.ndarray (lat, lon)
    N_MAP : np.ndarray (lat, lon)
    use_spatial : bool
    """
    log.info('── Section 3b: Pixel-wise IDF parameter fitting ───────────────')

    a_out  = os.path.join(OUTPUT_DIR, 'idf_param_a.nc')
    n_out  = os.path.join(OUTPUT_DIR, 'idf_param_n.nc')
    qc_out = os.path.join(OUTPUT_DIR, 'idf_param_fit_quality.nc')

    if not OVERWRITE_PARAMS and os.path.exists(a_out) and os.path.exists(n_out):
        log.info('  Parameter rasters already exist – loading them.')
        A_MAP = xr.open_dataset(a_out)['a'].values
        N_MAP = xr.open_dataset(n_out)['n'].values
        return A_MAP, N_MAP, True

    # ── 1. Load all IDF files ─────────────────────────────────────────────────
    log.info('  Loading IDF return-value rasters …')
    rv_list = []
    lat_ref = lon_ref = None

    for dur in ALL_DURATIONS:
        idf_path = os.path.join(OUTPUT_DIR, f'idf_{dur}h.nc')
        if not os.path.exists(idf_path):
            raise FileNotFoundError(
                f'{idf_path} not found.\nRun more_idf_italy.py first.'
            )
        ds = xr.open_dataset(idf_path)
        rv_list.append(
            ds['return_value'].sel(return_period=ALL_RETURN_PERIODS)
            .values.astype(np.float32)
        )
        if lat_ref is None:
            lat_ref = ds['lat'].values
            lon_ref = ds['lon'].values
        ds.close()
        log.info('    %dh loaded', dur)

    # (n_rp, n_dur, n_lat, n_lon)
    rv = np.stack(rv_list, axis=1)
    n_rp, n_dur, n_lat, n_lon = rv.shape
    durations_arr = np.array(ALL_DURATIONS, dtype=np.float32)
    log.info('  Return-value cube: %s  (return_periods × durations × lat × lon)', rv.shape)

    # ── 2. Intensity [mm/hr] ──────────────────────────────────────────────────
    intensities = rv / durations_arr[None, :, None, None]

    # ── 3. Vectorised OLS in log-log space ───────────────────────────────────
    log.info('  Fitting log-linear IDF (vectorised OLS) …')
    X    = np.log(durations_arr + B_FIX)
    Xbar = X.mean()
    Xc   = X - Xbar
    Xc2  = (Xc ** 2).sum()

    logY    = np.log(np.clip(intensities, 1e-9, None))
    logYbar = logY.mean(axis=1, keepdims=True)
    logYc   = logY - logYbar

    slopes     = (Xc[None, :, None, None] * logYc).sum(axis=1) / Xc2
    intercepts = logYbar[:, 0, :, :] - slopes * Xbar

    n_per_rp = -slopes
    a_per_rp = np.exp(intercepts)

    # ── 4. Summarise across return periods ───────────────────────────────────
    A_MAP = np.median(a_per_rp, axis=0).astype(np.float32)
    N_MAP = np.median(n_per_rp, axis=0).astype(np.float32)

    log.info(
        '  a: min=%.2f  max=%.2f  mean=%.2f mm/hr',
        np.nanmin(A_MAP), np.nanmax(A_MAP), np.nanmean(A_MAP)
    )
    log.info(
        '  n: min=%.3f  max=%.3f  mean=%.3f',
        np.nanmin(N_MAP), np.nanmax(N_MAP), np.nanmean(N_MAP)
    )

    # ── 5. (Optional) R² and RMSE ────────────────────────────────────────────
    if SAVE_FIT_QUALITY:
        logY_pred = (
            intercepts[:, None, :, :] + slopes[:, None, :, :] * X[None, :, None, None]
        )
        ss_res    = ((logY - logY_pred) ** 2).sum(axis=1)
        ss_tot    = ((logY - logYbar) ** 2).sum(axis=1)
        r2_per_rp = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
        r2_map    = np.nanmean(r2_per_rp, axis=0).astype(np.float32)
        rmse_map  = np.nanmean(np.sqrt(ss_res / n_dur), axis=0).astype(np.float32)
        log.info(
            '  R²: min=%.4f  mean=%.4f  pct<0.95=%.1f%%',
            np.nanmin(r2_map), np.nanmean(r2_map),
            float((r2_map < 0.95).mean() * 100)
        )

    # ── 6. Save ───────────────────────────────────────────────────────────────
    coords = {'lat': lat_ref, 'lon': lon_ref}
    enc    = {'dtype': 'float32', 'zlib': True, 'complevel': 4}

    xr.Dataset(
        {'a': xr.DataArray(
            A_MAP, dims=['lat', 'lon'], coords=coords,
            attrs={'units': 'mm/hr',
                   'long_name': 'IDF scale parameter a (i=a/(t+b)^n)',
                   'b_fixed': B_FIX}
        )},
        attrs={'fit': 'log-linear OLS over all durations × return periods',
               'b_fixed_hr': B_FIX}
    ).to_netcdf(a_out, engine='netcdf4', encoding={'a': enc})
    log.info('  Saved → %s', a_out)

    xr.Dataset(
        {'n': xr.DataArray(
            N_MAP, dims=['lat', 'lon'], coords=coords,
            attrs={'units': '-',
                   'long_name': 'IDF decay exponent n (i=a/(t+b)^n)',
                   'b_fixed': B_FIX}
        )},
        attrs={'fit': 'log-linear OLS over all durations × return periods',
               'b_fixed_hr': B_FIX}
    ).to_netcdf(n_out, engine='netcdf4', encoding={'n': enc})
    log.info('  Saved → %s', n_out)

    if SAVE_FIT_QUALITY:
        xr.Dataset({
            'r2': xr.DataArray(
                r2_map, dims=['lat', 'lon'], coords=coords,
                attrs={'units': '-', 'long_name': 'Mean R² of log-log IDF fit'}
            ),
            'rmse_log': xr.DataArray(
                rmse_map, dims=['lat', 'lon'], coords=coords,
                attrs={'units': 'log(mm/hr)', 'long_name': 'Mean RMSE in log space'}
            ),
        }).to_netcdf(qc_out, engine='netcdf4',
                     encoding={'r2': enc, 'rmse_log': enc})
        log.info('  Saved → %s', qc_out)

    log.info('  Pixel-wise IDF parameters ready.')
    return A_MAP, N_MAP, True


def load_or_skip_spatial_params():
    """
    Try to load existing pixel-wise IDF parameter rasters.
    Falls back to scalar CHICAGO_A / CHICAGO_N if files are absent.

    Returns (A_MAP or None, N_MAP or None, use_spatial: bool)
    """
    a_path = os.path.join(OUTPUT_DIR, 'idf_param_a.nc')
    n_path = os.path.join(OUTPUT_DIR, 'idf_param_n.nc')
    if os.path.exists(a_path) and os.path.exists(n_path):
        A_MAP = xr.open_dataset(a_path)['a'].values
        N_MAP = xr.open_dataset(n_path)['n'].values
        log.info('  Pixel-wise a, n loaded from existing files.')
        return A_MAP, N_MAP, True
    log.info('  Parameter rasters not found – using scalar CHICAGO_A=%.2f, CHICAGO_N=%.3f.',
             CHICAGO_A, CHICAGO_N)
    return None, None, False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — COMPUTE CHICAGO PEAK-INTENSITY MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_peak_maps(idf_data, A_MAP, N_MAP, use_spatial):
    """
    For each duration, mask unphysical depths, choose IDF parameters, and
    compute the Chicago peak-intensity map.

    Returns
    -------
    peak_maps   : dict {dur: np.ndarray (lat, lon)}
    hyeto_cubes : dict {dur: np.ndarray (T, lat, lon)}
    time_axes   : dict {dur: np.ndarray (T,)}
    """
    log.info('── Section 4: Computing Chicago peak-intensity maps ────────────')

    peak_maps   = {}
    hyeto_cubes = {}
    time_axes   = {}

    for dur in DURATIONS:
        depth_2d = idf_data[dur].values.astype(np.float32)

        bad = (
            ~np.isfinite(depth_2d) |
            (depth_2d < 0) |
            (depth_2d > MAX_PHYSICAL_DEPTH[dur])
        )
        depth_2d[bad] = np.nan

        if not np.any(np.isfinite(depth_2d)):
            log.warning(
                '  %dh: all pixels masked — check IDF raster or MAX_PHYSICAL_DEPTH cap.',
                dur
            )
            continue

        # Choose IDF parameters
        if use_spatial and A_MAP is not None:
            a_use = float(np.nanmedian(A_MAP))
            n_use = float(np.nanmedian(N_MAP))
            if not np.isfinite(a_use) or not np.isfinite(n_use):
                log.warning(
                    '  %dh: A_MAP/N_MAP are all NaN — falling back to scalar params. '
                    'Rerun param fitting with ALL_DURATIONS=%s', dur, ALL_DURATIONS
                )
                a_use = CHICAGO_A
                n_use = CHICAGO_N
            else:
                log.info('  %dh: using spatial a/n (median: a=%.3f  n=%.3f)',
                         dur, a_use, n_use)
        else:
            a_use = CHICAGO_A
            n_use = CHICAGO_N

        log.info(
            '  %dh depth_2d: finite=%d  nan=%d  min=%.2f  max=%.2f',
            dur,
            int(np.sum(np.isfinite(depth_2d))),
            int(np.sum(~np.isfinite(depth_2d))),
            float(np.nanmin(depth_2d)),
            float(np.nanmax(depth_2d))
        )
        log.info('  Computing Chicago peak map for %dh …', dur)

        peak, cube, t_min = chicago_peak_map(
            depth_2d,
            duration_hours=dur,
            dt_minutes=CHICAGO_DT,
            r=CHICAGO_R,
            a=a_use,
            b=CHICAGO_B,
            n=n_use
        )

        peak_maps[dur]   = peak
        hyeto_cubes[dur] = cube
        time_axes[dur]   = t_min

        valid = peak[np.isfinite(peak)]
        if valid.size == 0:
            log.warning('  %dh: no finite peak values.', dur)
        else:
            log.info(
                '  %dh peak intensity: min=%.1f  max=%.1f  mean=%.1f mm/hr',
                dur, valid.min(), valid.max(), valid.mean()
            )

    return peak_maps, hyeto_cubes, time_axes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SINGLE-PIXEL EXAMPLE HYETOGRAPHS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_example_hyetographs(idf_data, chicago_dir, target_rp):
    """
    Plot example Chicago hyetographs for POINT_NAME and save PNG.
    Output: <chicago_dir>/chicago_hyetograph_<POINT_NAME>_RP<rp>.png
    """
    log.info('── Section 5: Single-pixel example hyetographs ─────────────────')

    fig, axes = plt.subplots(1, len(DURATIONS), figsize=(6 * len(DURATIONS), 5),
                              sharey=False)
    if len(DURATIONS) == 1:
        axes = [axes]

    for ax, dur in zip(axes, DURATIONS):
        depth_pt = float(
            idf_data[dur].sel(lat=POINT_LAT, lon=POINT_LON, method='nearest')
        )
        df_hyet = create_chicago_hyetograph(
            total_precip_mm=depth_pt,
            duration_hours=dur,
            dt_minutes=CHICAGO_DT,
            r=CHICAGO_R, a=CHICAGO_A, b=CHICAGO_B, n=CHICAGO_N
        )
        ax.bar(
            df_hyet['time_min'], df_hyet['intensity_mm_hr'],
            width=CHICAGO_DT * 0.9, align='edge',
            color='steelblue', alpha=0.8, label=f'Δt = {CHICAGO_DT:.0f} min'
        )
        peak_i = df_hyet['intensity_mm_hr'].max()
        peak_t = df_hyet.loc[df_hyet['intensity_mm_hr'].idxmax(), 'time_min']
        ax.axvline(peak_t + CHICAGO_DT / 2, color='red', ls='--', lw=1.2,
                   label=f'Peak {peak_i:.1f} mm/hr @ {peak_t:.0f} min')
        ax.set_xlabel('Time [min]', fontsize=11)
        ax.set_ylabel('Intensity [mm/hr]', fontsize=11)
        ax.set_title(
            f'Chicago hyetograph – {dur}h – RP{target_rp}\n'
            f'{POINT_NAME} ({POINT_LAT}°N, {POINT_LON}°E)  |  '
            f'Total depth: {depth_pt:.1f} mm',
            fontsize=10
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hyet_png = os.path.join(
        chicago_dir,
        f'chicago_hyetograph_{POINT_NAME.replace(" ", "_")}_RP{target_rp}.png'
    )
    fig.savefig(hyet_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved → %s', hyet_png)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SPATIAL MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_map(ax, data, lat, lon, title, cmap, vmin, vmax, label):
    """Thin wrapper: plot with cartopy or plain imshow depending on availability."""
    if HAS_CARTOPY:
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                      crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, linestyle=':')
        ax.add_feature(cfeature.LAND,      facecolor='#f5f5f0', zorder=0)
        im = ax.pcolormesh(
            lon, lat, data,
            cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(), zorder=1
        )
    else:
        im = ax.imshow(
            data, origin='lower',
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
            cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto'
        )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    return im


def plot_peak_intensity_maps(idf_data, peak_maps, chicago_dir, target_rp):
    """
    6.1 One figure with one subplot per duration: Chicago peak intensity maps.
    Output: <chicago_dir>/chicago_peak_intensity_RP<rp>.png

    6.2 Side-by-side IDF depth vs. peak intensity for the first duration.
    Output: <chicago_dir>/chicago_depth_vs_peak_<dur>h_RP<rp>.png
    """
    log.info('── Section 6: Spatial maps ─────────────────────────────────────')

    lat_arr = idf_data[DURATIONS[0]]['lat'].values
    lon_arr = idf_data[DURATIONS[0]]['lon'].values
    log.info(
        'Grid: %d × %d  lat [%.2f, %.2f]  lon [%.2f, %.2f]',
        len(lat_arr), len(lon_arr),
        lat_arr.min(), lat_arr.max(),
        lon_arr.min(), lon_arr.max()
    )

    # ── 6.1 Peak intensity maps ───────────────────────────────────────────────
    n_cols = len(DURATIONS)
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            1, n_cols, figsize=(7 * n_cols, 7),
            subplot_kw={'projection': proj}
        )
    else:
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))

    if n_cols == 1:
        axes = [axes]

    for ax, dur in zip(axes, DURATIONS):
        peak = peak_maps[dur].copy()
        peak[peak == 0] = np.nan
        valid  = peak[np.isfinite(peak)]
        vmin_p = np.nanpercentile(valid, 2)
        vmax_p = np.nanpercentile(valid, 98)

        _make_map(
            ax, peak, lat_arr, lon_arr,
            title=(
                f'Chicago peak intensity — {dur}h / RP{target_rp}\n'
                f'r={CHICAGO_R}, a={CHICAGO_A}, b={CHICAGO_B}, n={CHICAGO_N}'
            ),
            cmap='YlOrRd', vmin=vmin_p, vmax=vmax_p,
            label='Peak intensity [mm/hr]'
        )

        if HAS_CARTOPY:
            ax.plot(POINT_LON, POINT_LAT, 'k^', ms=5,
                    transform=ccrs.PlateCarree(), zorder=5, label=POINT_NAME)
        else:
            ax.plot(POINT_LON, POINT_LAT, 'k^', ms=5, label=POINT_NAME)
        ax.legend(fontsize=8, loc='lower right')

    plt.suptitle(
        f'Chicago Hyetograph — Peak Intensity Maps — RP{target_rp}',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out_png = os.path.join(chicago_dir, f'chicago_peak_intensity_RP{target_rp}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved → %s', out_png)

    # ── 6.2 IDF depth vs. peak intensity (first duration) ────────────────────
    dur_check = DURATIONS[0]

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 7),
            subplot_kw={'projection': proj}
        )
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    depth_2d = idf_data[dur_check].values.copy()
    depth_2d[depth_2d == 0] = np.nan
    _make_map(
        ax1, depth_2d, lat_arr, lon_arr,
        title=f'IDF total depth — {dur_check}h / RP{target_rp}',
        cmap='Blues',
        vmin=np.nanpercentile(depth_2d, 2),
        vmax=np.nanpercentile(depth_2d, 98),
        label='Depth [mm]'
    )

    peak_check = peak_maps[dur_check].copy()
    peak_check[peak_check == 0] = np.nan
    finite_peak = peak_check[np.isfinite(peak_check)]
    _make_map(
        ax2, peak_check, lat_arr, lon_arr,
        title=f'Chicago peak intensity — {dur_check}h / RP{target_rp}',
        cmap='YlOrRd',
        vmin=np.nanpercentile(finite_peak, 2),
        vmax=np.nanpercentile(finite_peak, 98),
        label='Peak intensity [mm/hr]'
    )

    plt.suptitle(
        f'{dur_check}h Design Storm — RP{target_rp}  (Chicago, r={CHICAGO_R})',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    side_png = os.path.join(
        chicago_dir,
        f'chicago_depth_vs_peak_{dur_check}h_RP{target_rp}.png'
    )
    fig.savefig(side_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info('Saved → %s', side_png)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EXPORT TO NETCDF
# ═══════════════════════════════════════════════════════════════════════════════

def export_netcdf(idf_data, peak_maps, hyeto_cubes, time_axes, chicago_dir, target_rp):
    """
    Save one NetCDF per duration containing peak_intensity and total_depth.
    Optionally includes the full hyetograph cube if SAVE_CUBE is True.
    Output: <chicago_dir>/chicago_<dur>h_RP<rp>.nc
    """
    log.info('── Section 7: Exporting NetCDF ─────────────────────────────────')

    lat_arr = idf_data[DURATIONS[0]]['lat'].values
    lon_arr = idf_data[DURATIONS[0]]['lon'].values

    for dur in DURATIONS:
        peak  = peak_maps[dur]
        depth = idf_data[dur].values
        t_min = time_axes[dur]

        ds_out = xr.Dataset(
            {
                'peak_intensity': xr.DataArray(
                    peak, dims=['lat', 'lon'],
                    coords={'lat': lat_arr, 'lon': lon_arr},
                    attrs={
                        'units': 'mm/hr',
                        'long_name': f'Chicago peak intensity {dur}h RP{target_rp}',
                        'chicago_r': CHICAGO_R, 'idf_a': CHICAGO_A,
                        'idf_b': CHICAGO_B, 'idf_n': CHICAGO_N,
                        'dt_min': CHICAGO_DT,
                    }
                ),
                'total_depth': xr.DataArray(
                    depth, dims=['lat', 'lon'],
                    coords={'lat': lat_arr, 'lon': lon_arr},
                    attrs={'units': 'mm',
                           'long_name': f'GEV return depth {dur}h RP{target_rp}'}
                ),
            },
            attrs={
                'description': 'Chicago (Keifer & Chu) hyetograph applied to MORE IDF rasters',
                'return_period_years': target_rp,
                'duration_hours': dur,
            }
        )

        if SAVE_CUBE:
            ds_out['hyetograph'] = xr.DataArray(
                hyeto_cubes[dur],
                dims=['time_min', 'lat', 'lon'],
                coords={'time_min': t_min, 'lat': lat_arr, 'lon': lon_arr},
                attrs={'units': 'mm/hr', 'long_name': 'Chicago intensity time series'}
            )

        nc_path = os.path.join(chicago_dir, f'chicago_{dur}h_RP{target_rp}.nc')
        ds_out.to_netcdf(nc_path, engine='netcdf4', encoding={
            'peak_intensity': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
            'total_depth':    {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        })
        log.info('  Saved → %s', nc_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def log_summary(idf_data, peak_maps, target_rp):
    """Log a per-duration summary table of depth and peak intensity."""
    log.info('── Section 8: Summary statistics ──────────────────────────────')
    log.info('Chicago hyetograph summary — RP%d', target_rp)
    log.info(
        'Parameters: r=%s  a=%s  b=%s  n=%s  Δt=%s min',
        CHICAGO_R, CHICAGO_A, CHICAGO_B, CHICAGO_N, CHICAGO_DT
    )
    header = (
        f'{"Duration":>10s}{"Depth min":>12s}{"Depth max":>12s}'
        f'{"Peak i min":>12s}{"Peak i max":>12s}{"Peak i mean":>13s}'
    )
    log.info('%s', header)
    log.info('%s', '─' * len(header))
    for dur in DURATIONS:
        depth = idf_data[dur].values
        peak  = peak_maps[dur]
        mask  = np.isfinite(peak) & (peak > 0)
        log.info(
            '%s%s%s%s%s%s',
            f'{str(dur)+"h":>10s}',
            f'{np.nanmin(depth):12.1f}',
            f'{np.nanmax(depth):12.1f}',
            f'{peak[mask].min():12.1f}',
            f'{peak[mask].max():12.1f}',
            f'{peak[mask].mean():13.1f}',
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='MORE dataset: Chicago hyetograph maps for Italy (1991–2020)'
    )
    p.add_argument('--output-dir', default=OUTPUT_DIR,
                   help='Folder containing IDF NetCDF files (idf_<d>h.nc)')
    p.add_argument('--target-rp',  type=int, default=TARGET_RP,
                   help='Return period in years (default: %(default)s)')
    p.add_argument('--chicago-r',  type=float, default=CHICAGO_R,
                   help='Chicago peak-position ratio (default: %(default)s)')
    p.add_argument('--chicago-dt', type=float, default=CHICAGO_DT,
                   help='Hyetograph time step in minutes (default: %(default)s)')
    p.add_argument('--skip-plots', action='store_true',
                   help='Skip all matplotlib output (headless / batch mode)')
    p.add_argument('--skip-inspect', action='store_true',
                   help='Skip the IDF data inspection step')
    p.add_argument('--log-file',   default=LOG_FILE,
                   help='Optional path for a persistent log file')
    return p.parse_args()


def main():
    global OUTPUT_DIR, TARGET_RP, CHICAGO_R, CHICAGO_DT, LOG_FILE

    args       = _parse_args()
    OUTPUT_DIR = args.output_dir
    TARGET_RP  = args.target_rp
    CHICAGO_R  = args.chicago_r
    CHICAGO_DT = args.chicago_dt
    LOG_FILE   = args.log_file

    _setup_logging(LOG_FILE)

    chicago_dir = os.path.join(OUTPUT_DIR, 'chicago')
    os.makedirs(chicago_dir, exist_ok=True)

    log.info('════════════════════════════════════════════════════════════════')
    log.info('MORE Chicago hyetograph analysis — start')
    log.info('  OUTPUT_DIR  : %s', OUTPUT_DIR)
    log.info('  CHICAGO_DIR : %s', chicago_dir)
    log.info('  Target RP   : %d years', TARGET_RP)
    log.info('  Durations   : %s h', DURATIONS)
    log.info('  Chicago r   : %.2f  a=%.2f  b=%.2f  n=%.2f  Δt=%.0f min',
             CHICAGO_R, CHICAGO_A, CHICAGO_B, CHICAGO_N, CHICAGO_DT)
    log.info('  cartopy     : %s', HAS_CARTOPY)
    log.info('════════════════════════════════════════════════════════════════')

    # ── 0. Inspection report ──────────────────────────────────────────────────
    if not args.skip_inspect:
        inspect_idf_data(chicago_dir)

    # ── 3. Load IDF rasters ───────────────────────────────────────────────────
    idf_data = load_idf_data(TARGET_RP)

    # ── 3b. Pixel-wise IDF parameter fitting ─────────────────────────────────
    A_MAP, N_MAP, use_spatial = fit_idf_parameters()

    # ── 4. Compute Chicago peak maps ─────────────────────────────────────────
    peak_maps, hyeto_cubes, time_axes = compute_peak_maps(
        idf_data, A_MAP, N_MAP, use_spatial
    )

    # ── 5–6. Plots ────────────────────────────────────────────────────────────
    if not args.skip_plots:
        plot_example_hyetographs(idf_data, chicago_dir, TARGET_RP)
        plot_peak_intensity_maps(idf_data, peak_maps, chicago_dir, TARGET_RP)
    else:
        log.info('--skip-plots set: matplotlib output suppressed.')

    # ── 7. NetCDF export ──────────────────────────────────────────────────────
    export_netcdf(idf_data, peak_maps, hyeto_cubes, time_axes, chicago_dir, TARGET_RP)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    log_summary(idf_data, peak_maps, TARGET_RP)

    log.info('MORE Chicago hyetograph analysis — complete.')


if __name__ == '__main__':
    main()
