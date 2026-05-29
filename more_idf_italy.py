#!/usr/bin/env python3
"""
more_idf_italy.py
=================
MORE Dataset: Precipitation IDF Risk Maps for Italy
MOloch-downscaled ERA5 REanalysis (1991–2020) — Return Period Analysis

Dataset  : https://doi.org/10.5281/zenodo.18470948
Spatial  : ~1.7 km  |  Temporal: hourly
Variable : moloch_tp_* — total precipitation
Goal     : IDF (Intensity–Duration–Frequency) maps for RP = 5, 10, 25, 50, 100 years
Durations: 1h, 3h, 6h, 12h, 24h

Usage
-----
    python more_idf_italy.py [--data-root PATH] [--output-dir PATH]
                             [--years 1991-2020] [--skip-plots]
                             [--log-file PATH]

All settings can also be edited in the USER SETTINGS block below.
"""

# ── standard library ──────────────────────────────────────────────────────────
import argparse
import logging
import os
import sys

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for batch jobs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors                   # noqa: F401  (kept for completeness)
from matplotlib.colors import from_levels_and_colors
from scipy.stats import genextreme, ks_1samp
from scipy.stats import probplot
from tqdm.auto import tqdm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── local helper (memory-efficient annual-max) ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import idf_chunked_helpers as _h
rolling_accumulation_annual_max_chunked = _h.rolling_accumulation_annual_max_chunked


# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS  (override via CLI or environment variables)
# ═══════════════════════════════════════════════════════════════════════════════

DATA_ROOT   = os.environ.get('MORE_DATA_ROOT',   r'/mnt/data/more')
OUTPUT_DIR  = os.environ.get('MORE_OUTPUT_DIR',
              os.path.join('/home/admin_climatecharted_com/data/MOloch', 'IDF_results'))
LOG_FILE    = os.environ.get('MORE_LOG_FILE', None)     # None → stderr only

YEARS           = list(range(1991, 2021))               # 1991–2020
TP_VAR          = 'tp'                                  # adjust if needed
DURATIONS       = [1, 3, 6, 12, 24]                    # hours
RETURN_PERIODS  = np.array([5, 10, 25, 50, 100])        # years

# Validation point
POINT_NAME = 'Milan'
POINT_LAT  = 45.46
POINT_LON  = 9.19
VALIDATE_DUR = 1                                        # duration used for GEV validation plot

# GEV stability cap  (ξ = -c in scipy convention; cap at 0.5 per ISPRA practice)
GEV_XI_CAP = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_logging(log_file=None):
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers)

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATASET INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_dataset(test_file):
    """
    Open one monthly file, log its structure and basic statistics, and
    auto-detect whether de-accumulation is required.

    Returns
    -------
    tp_var     : str   – precipitation variable name to use
    deacc      : bool  – True if de-accumulation is needed
    """
    log.info('── Section 2: Dataset inspection ──────────────────────────────')
    log.info('Opening test file: %s', test_file)

    ds = xr.open_dataset(test_file)
    log.info('Dataset overview:\n%s', ds)

    # 2.2  Variable / coordinate inventory
    log.info('=== VARIABLES ===')
    for vname, var in ds.data_vars.items():
        log.info('  %-30s  dims=%s  dtype=%s  units=%s',
                 vname, var.dims, var.dtype, var.attrs.get('units', '?'))

    log.info('=== COORDINATES ===')
    for cname, coord in ds.coords.items():
        log.info('  %-20s  shape=%s', cname, coord.shape)

    log.info('=== GLOBAL ATTRIBUTES ===')
    for k, v in ds.attrs.items():
        log.info('  %s: %s', k, v)

    # 2.3  Auto-detect precipitation variable
    candidates = [v for v in ds.data_vars
                  if any(kw in v.lower() for kw in ('tp', 'prec', 'rain'))]
    tp_var = candidates[0] if candidates else TP_VAR
    if candidates:
        log.info('Using precipitation variable: "%s"', tp_var)
    else:
        log.warning('Could not auto-detect variable; using TP_VAR="%s". '
                    'Available: %s', tp_var, list(ds.data_vars))

    # 2.4  Time axis
    times = ds.time.values
    log.info('Time steps : %d', len(times))
    log.info('Start      : %s', times[0])
    log.info('End        : %s', times[-1])
    if len(times) > 1:
        dt_h = (times[1] - times[0]) / np.timedelta64(1, 'h')
        log.info('Time step  : %.1f h', dt_h)

    # 2.5  Spatial coverage
    lat = ds['lat'].values if 'lat' in ds.coords else ds['latitude'].values
    lon = ds['lon'].values if 'lon' in ds.coords else ds['longitude'].values
    log.info('Lat range  : %.3f – %.3f', lat.min(), lat.max())
    log.info('Lon range  : %.3f – %.3f', lon.min(), lon.max())
    log.info('Grid shape : %s', lat.shape)

    # 2.6  Basic statistics (first 24 h only for speed)
    da_tp = ds[tp_var].isel(time=slice(0, 24))
    log.info('Shape (time, lat, lon): %s', da_tp.shape)
    log.info('Min  : %s', float(da_tp.min()))
    log.info('Max  : %s', float(da_tp.max()))
    log.info('Mean : %s', float(da_tp.mean()))
    log.info('Units: %s', da_tp.attrs.get('units', 'unknown'))

    # 2.7  Quick snapshot map
    da_snap = ds[tp_var].isel(time=0).squeeze()
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(lon, lat, da_snap.values, cmap='Blues', vmin=0)
    plt.colorbar(im, ax=ax, label=f'Precipitation [{da_tp.attrs.get("units", "mm")}]')
    ax.set_title(f'MORE – {tp_var} – {str(times[0])[:16]}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    snap_png = os.path.join(OUTPUT_DIR, 'snapshot_202005_t0.png')
    fig.savefig(snap_png, dpi=150)
    plt.close(fig)
    log.info('Snapshot saved → %s', snap_png)

    # 2.8  Central-pixel time series
    mid_lat = lat.shape[0] // 2
    mid_lon = lon.shape[-1] // 2
    ts = ds[tp_var].isel(lat=mid_lat, lon=mid_lon).squeeze()
    fig, ax = plt.subplots(figsize=(14, 4))
    ts.plot(ax=ax)
    ax.set_title('Hourly precipitation time series – central pixel (May 2020)')
    plt.tight_layout()
    ts_png = os.path.join(OUTPUT_DIR, 'timeseries_central_pixel_202005.png')
    fig.savefig(ts_png, dpi=150)
    plt.close(fig)
    log.info('Time-series plot saved → %s', ts_png)

    # 2.9  Detect accumulation type
    sample = ds[tp_var].isel(lat=mid_lat, lon=mid_lon).squeeze().values
    n_neg = (np.diff(sample[:48]) < -0.01).sum()
    log.info('Negative differences in first 48 h: %d', n_neg)
    if n_neg > 5:
        log.info('→ Variable appears to be HOURLY AMOUNTS. No de-accumulation needed.')
        deacc = False
    else:
        log.info('→ Variable may be CUMULATIVE from run start. De-accumulation will be applied.')
        deacc = True

    ds.close()
    return tp_var, deacc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HELPER FUNCTIONS  (GEV fitting)
# ═══════════════════════════════════════════════════════════════════════════════

def fit_gev_pixel(ts):
    """
    Fit a GEV distribution to a 1-D array of annual maxima.
    Returns (shape, loc, scale) or NaN-tuple on failure.
    Shape parameter ξ is capped at GEV_XI_CAP.
    scipy convention: shape c = -ξ; cap → c ≥ -GEV_XI_CAP.
    """
    valid = ts[~np.isnan(ts)]
    if len(valid) < 5 or valid.max() == 0:
        return (np.nan, np.nan, np.nan)
    try:
        c, loc, scale = genextreme.fit(valid)
        c = max(c, -GEV_XI_CAP)
        return (c, loc, scale)
    except Exception:
        return (np.nan, np.nan, np.nan)


def gev_return_value(params, return_period):
    """Compute the GEV return value for a given return period."""
    shape, loc, scale = params
    if np.isnan(shape):
        return np.nan
    return genextreme.ppf(1.0 - 1.0 / return_period, shape, loc=loc, scale=scale)


def fit_and_compute_rp(annual_max_stack, return_periods):
    """
    Apply GEV fitting and return-value calculation over the full spatial grid.

    Parameters
    ----------
    annual_max_stack : np.ndarray  (n_years, n_lat, n_lon)
    return_periods   : 1-D array

    Returns
    -------
    rv : np.ndarray  (n_rp, n_lat, n_lon)
    """
    n_years, n_lat, n_lon = annual_max_stack.shape
    n_rp = len(return_periods)
    rv = np.full((n_rp, n_lat, n_lon), np.nan, dtype=np.float32)

    for i in tqdm(range(n_lat), desc='  GEV fitting rows'):
        for j in range(n_lon):
            ts = annual_max_stack[:, i, j]
            params = fit_gev_pixel(ts)
            for k, rp in enumerate(return_periods):
                rv[k, i, j] = gev_return_value(params, rp)
    return rv


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD ANNUAL-MAXIMUM STACKS
# ═══════════════════════════════════════════════════════════════════════════════

def build_annual_maxima(tp_var, deacc):
    """
    For each duration in DURATIONS, compute and save annual-maximum stacks.
    Uses the memory-efficient chunked helper (~5 GB peak RAM per year).
    Output: <OUTPUT_DIR>/annmax_<dur>h.nc
    """
    log.info('── Section 4: Building annual-maximum stacks ──────────────────')
    log.info('Peak RAM strategy: chunked monthly processing (~5 GB/year)')

    for dur in DURATIONS:
        out_path = os.path.join(OUTPUT_DIR, f'annmax_{dur}h.nc')
        if os.path.exists(out_path):
            log.info('  %dh – already exists, skipping.', dur)
            continue

        log.info('Duration: %dh', dur)
        stacks = []

        for yr in tqdm(YEARS, desc=f'{dur}h annual max'):
            try:
                ann_max = rolling_accumulation_annual_max_chunked(
                    yr, DATA_ROOT, tp_var, dur, deacc=deacc
                )
            except FileNotFoundError as exc:
                log.warning('  Year %d – file not found, skipping. Detail: %s', yr, exc)
                continue

            ann_max = ann_max.expand_dims(dim={'year': [yr]})
            stacks.append(ann_max)

        if not stacks:
            log.warning('  No data for duration %dh – skipping.', dur)
            continue

        ds_stack = xr.concat(stacks, dim='year')
        ds_stack.name = f'annmax_tp_{dur}h'
        ds_stack.attrs['units']       = 'mm'
        ds_stack.attrs['description'] = (
            f'Annual maximum {dur}h precipitation (MORE dataset)'
        )
        ds_stack.to_dataset(name=f'annmax_tp_{dur}h').to_netcdf(out_path)
        log.info('  Saved → %s', out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GEV FITTING & RETURN-VALUE MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_geotiff(array, lat, lon, out_path):
    """Write a 2-D float32 array to a GeoTIFF (EPSG:4326, LZW compressed)."""
    if not HAS_RASTERIO:
        log.warning('  rasterio not available – skipping GeoTIFF export.')
        return

    arr = array.astype('float32')

    if lat[0] < lat[-1]:          # ascending → flip for north-up raster
        arr = arr[::-1, :]
        lat_top, lat_bottom = float(lat[-1]), float(lat[0])
    else:
        lat_top, lat_bottom = float(lat[0]), float(lat[-1])

    dlon = float(lon[1] - lon[0])
    dlat = abs(float(lat[1] - lat[0]))
    west  = float(lon[0])  - dlon / 2
    east  = float(lon[-1]) + dlon / 2
    north = lat_top    + dlat / 2
    south = lat_bottom - dlat / 2

    transform = from_bounds(west, south, east, north, arr.shape[1], arr.shape[0])

    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=arr.shape[0], width=arr.shape[1],
        count=1, dtype='float32',
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress='lzw',
        nodata=float('nan'),
    ) as dst:
        dst.write(arr, 1)


def compute_idf_return_values():
    """
    For each duration, load the annual-max stack, fit GEV per pixel, compute
    return values, save as NetCDF and individual GeoTIFFs.
    Output: <OUTPUT_DIR>/idf_<dur>h.nc  +  idf_<dur>h_RP<rp>.tif
    """
    log.info('── Section 5: GEV fitting & return-value computation ──────────')

    for dur in DURATIONS:
        in_path  = os.path.join(OUTPUT_DIR, f'annmax_{dur}h.nc')
        out_path = os.path.join(OUTPUT_DIR, f'idf_{dur}h.nc')

        if not os.path.exists(in_path):
            log.warning('  %dh – annual max file missing, skipping.', dur)
            continue

        if os.path.exists(out_path):
            log.info('  %dh – IDF file already exists, skipping.', dur)
            # Still export any missing GeoTIFFs
            ds_idf = xr.open_dataset(out_path)
            lat_c = ds_idf['lat'].values
            lon_c = ds_idf['lon'].values
            for rp in RETURN_PERIODS:
                tif_path = os.path.join(OUTPUT_DIR, f'idf_{dur}h_RP{rp}.tif')
                if not os.path.exists(tif_path):
                    rv_slice = ds_idf['return_value'].sel(return_period=rp).values
                    _write_geotiff(rv_slice, lat_c, lon_c, tif_path)
                    log.info('    GeoTIFF → %s', tif_path)
            ds_idf.close()
            continue

        log.info('GEV fitting for %dh', dur)
        ds_am   = xr.open_dataset(in_path)
        varname = f'annmax_tp_{dur}h'
        stack   = ds_am[varname].values          # (n_years, n_lat, n_lon)

        rv = fit_and_compute_rp(stack, RETURN_PERIODS)   # (n_rp, n_lat, n_lon)

        lat_coord = ds_am['lat'] if 'lat' in ds_am.coords else ds_am['latitude']
        lon_coord = ds_am['lon'] if 'lon' in ds_am.coords else ds_am['longitude']

        ds_idf = xr.Dataset(
            {'return_value': (['return_period', 'lat', 'lon'],
                              rv,
                              {'units': 'mm',
                               'long_name': f'{dur}h precipitation return value'})},
            coords={
                'return_period': RETURN_PERIODS,
                'lat': lat_coord,
                'lon': lon_coord,
            },
            attrs={
                'duration_h': dur,
                'distribution': 'GEV (scipy genextreme)',
                'source': 'MORE v1.0 – 1991-2020 hourly precipitation',
            }
        )
        ds_idf.to_netcdf(out_path, engine='netcdf4')
        log.info('  Saved → %s', out_path)

        lat_vals = lat_coord.values if hasattr(lat_coord, 'values') else np.array(lat_coord)
        lon_vals = lon_coord.values if hasattr(lon_coord, 'values') else np.array(lon_coord)
        for i_rp, rp in enumerate(RETURN_PERIODS):
            tif_path = os.path.join(OUTPUT_DIR, f'idf_{dur}h_RP{rp}.tif')
            _write_geotiff(rv[i_rp], lat_vals, lon_vals, tif_path)
            log.info('  GeoTIFF → %s', tif_path)

        ds_am.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — IDF MAPS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_precip_cmap():
    levels = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 250, 300, 400]
    colors = [
        '#ffffff', '#d6e2ff', '#8db2ff', '#626ff7', '#0062ff',
        '#019696', '#01c634', '#63ff01', '#c6ff34', '#ffff02',
        '#ffc601', '#ffa001', '#ff7c00', '#ff1901',
        '#cc0000', '#990033', '#660066',
    ]
    return from_levels_and_colors(levels, colors, extend='max')


def plot_idf_maps():
    """
    Produce one PNG per duration showing return-value maps for all return periods.
    Output: <OUTPUT_DIR>/IDF_maps_<dur>h.png
    """
    log.info('── Section 6: Plotting IDF maps ───────────────────────────────')
    precip_cmap, precip_norm, _ = _make_precip_cmap()

    for dur in DURATIONS:
        idf_path = os.path.join(OUTPUT_DIR, f'idf_{dur}h.nc')
        if not os.path.exists(idf_path):
            log.warning('  %dh IDF file not found – skipping plots.', dur)
            continue

        ds_idf = xr.open_dataset(idf_path)
        lat_v  = ds_idf['lat'].values
        lon_v  = ds_idf['lon'].values

        n_rp  = len(RETURN_PERIODS)
        ncols = min(n_rp, 5)
        nrows = (n_rp + ncols - 1) // ncols

        if HAS_CARTOPY:
            proj = ccrs.PlateCarree()
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                subplot_kw={'projection': proj},
                constrained_layout=True
            )
        else:
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                constrained_layout=True
            )

        axes_flat = np.array(axes).flatten()
        fig.suptitle(
            f'Precipitation return values – {dur}h duration (MORE 1991–2020)',
            fontsize=14
        )

        for k, rp in enumerate(RETURN_PERIODS):
            ax = axes_flat[k]
            rv_map = ds_idf['return_value'].sel(return_period=rp).values

            if HAS_CARTOPY:
                im = ax.pcolormesh(
                    lon_v, lat_v, rv_map,
                    cmap=precip_cmap, norm=precip_norm,
                    transform=ccrs.PlateCarree()
                )
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
                ax.add_feature(cfeature.BORDERS, linewidth=0.4)
                ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', zorder=0)
                gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
                gl.top_labels = gl.right_labels = False
            else:
                im = ax.pcolormesh(lon_v, lat_v, rv_map,
                                   cmap=precip_cmap, norm=precip_norm)
                ax.set_xlabel('Lon')
                ax.set_ylabel('Lat')

            ax.set_title(f'RP = {rp} yr', fontsize=11)
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05,
                         label='mm', shrink=0.85)

        for k in range(len(RETURN_PERIODS), len(axes_flat)):
            axes_flat[k].set_visible(False)

        out_png = os.path.join(OUTPUT_DIR, f'IDF_maps_{dur}h.png')
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        log.info('  Saved → %s', out_png)
        ds_idf.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — IDF CURVES AT A POINT OF INTEREST
# ═══════════════════════════════════════════════════════════════════════════════

def plot_idf_curves():
    """
    Extract IDF values at POINT_NAME, log the table, and save an IDF-curve PNG.
    Output: <OUTPUT_DIR>/IDF_curves_<POINT_NAME>.png
    """
    log.info('── Section 7: IDF curves at %s (%.2f°N, %.2f°E) ──────────────',
             POINT_NAME, POINT_LAT, POINT_LON)

    idf_table = {}
    for dur in DURATIONS:
        idf_path = os.path.join(OUTPUT_DIR, f'idf_{dur}h.nc')
        if not os.path.exists(idf_path):
            continue
        ds_idf = xr.open_dataset(idf_path)
        point_data = ds_idf['return_value'].sel(
            lat=POINT_LAT, lon=POINT_LON, method='nearest'
        ).values
        idf_table[dur] = dict(zip(RETURN_PERIODS.tolist(), point_data.tolist()))
        ds_idf.close()

    if not idf_table:
        log.warning('  No IDF data found – skipping IDF-curve section.')
        return

    # Log table
    header = f'{"Duration":>10s}' + ''.join([f'{"RP"+str(rp):>10s}' for rp in RETURN_PERIODS])
    log.info('IDF table at %s (%s°N, %s°E)', POINT_NAME, POINT_LAT, POINT_LON)
    log.info('%s', header)
    log.info('%s', '─' * len(header))
    for dur, rp_vals in idf_table.items():
        row = f'{str(dur)+"h":>10s}' + ''.join([f'{v:10.1f}' for v in rp_vals.values()])
        log.info('%s', row)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ['o', 's', '^', 'D', 'v']
    for i, (dur, rp_vals) in enumerate(idf_table.items()):
        ax.plot(list(rp_vals.keys()), list(rp_vals.values()),
                marker=markers[i % len(markers)], label=f'{dur}h', linewidth=2)

    ax.set_xscale('log')
    ax.set_xticks(RETURN_PERIODS)
    ax.set_xticklabels(RETURN_PERIODS)
    ax.set_xlabel('Return Period (years)', fontsize=12)
    ax.set_ylabel('Precipitation (mm)', fontsize=12)
    ax.set_title(f'IDF Curves – {POINT_NAME} ({POINT_LAT}°N, {POINT_LON}°E)', fontsize=13)
    ax.legend(title='Duration', fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, f'IDF_curves_{POINT_NAME.replace(" ", "_")}.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    log.info('IDF curves saved → %s', out_png)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — GEV VALIDATION AT A POINT
# ═══════════════════════════════════════════════════════════════════════════════

def validate_gev():
    """
    Load the 1-h annual-maxima time series for POINT_NAME, fit GEV, log
    KS-test results, and save a 3-panel diagnostic figure.
    Output: <OUTPUT_DIR>/GEV_validation_<POINT_NAME>_<dur>h.png
    """
    log.info('── Section 8: GEV validation at %s (%dh) ──────────────────────',
             POINT_NAME, VALIDATE_DUR)

    am_path = os.path.join(OUTPUT_DIR, f'annmax_{VALIDATE_DUR}h.nc')
    if not os.path.exists(am_path):
        log.warning('Annual max file %s not found – run Section 4 first.', am_path)
        return

    ds_am   = xr.open_dataset(am_path)
    varname = f'annmax_tp_{VALIDATE_DUR}h'
    ts_point = ds_am[varname].sel(
        lat=POINT_LAT, lon=POINT_LON, method='nearest'
    ).values

    valid = ts_point[~np.isnan(ts_point)]
    params = genextreme.fit(valid)
    shape, loc, scale = params

    log.info('GEV parameters for %s, %dh:', POINT_NAME, VALIDATE_DUR)
    log.info('  shape (ξ) = %.4f', shape)
    log.info('  loc   (μ) = %.4f mm', loc)
    log.info('  scale (σ) = %.4f mm', scale)

    ks_stat, ks_p = ks_1samp(valid, genextreme.cdf, args=params)
    log.info('KS test: statistic=%.4f, p-value=%.4f', ks_stat, ks_p)
    log.info('Fit is %s', 'GOOD (p > 0.05)' if ks_p > 0.05 else 'POOR (p ≤ 0.05)')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Q-Q plot
    (osm, osr), (slope, intercept, _r) = probplot(
        valid, dist=genextreme, sparams=params
    )
    axes[0].plot(osm, osr, 'o', alpha=0.7)
    axes[0].plot([osm.min(), osm.max()],
                 [slope * osm.min() + intercept, slope * osm.max() + intercept], 'r-')
    axes[0].set_title('Q-Q plot')
    axes[0].set_xlabel('Theoretical')
    axes[0].set_ylabel('Observed')

    # Empirical vs. theoretical CDF
    x = np.linspace(valid.min(), valid.max(), 200)
    axes[1].plot(x, genextreme.cdf(x, *params), 'r-', label='GEV CDF')
    axes[1].hist(valid, density=True, cumulative=True, bins=10, alpha=0.5, label='Empirical')
    axes[1].set_title('CDF')
    axes[1].legend()

    # Annual-max bar chart
    axes[2].bar(YEARS[:len(ts_point)], ts_point, color='steelblue', alpha=0.7)
    axes[2].set_title(f'Annual max {VALIDATE_DUR}h – {POINT_NAME}')
    axes[2].set_xlabel('Year')
    axes[2].set_ylabel('mm')

    plt.suptitle(f'GEV validation – {VALIDATE_DUR}h – {POINT_NAME}', fontsize=12)
    plt.tight_layout()

    val_png = os.path.join(
        OUTPUT_DIR,
        f'GEV_validation_{POINT_NAME.replace(" ", "_")}_{VALIDATE_DUR}h.png'
    )
    fig.savefig(val_png, dpi=150)
    plt.close(fig)
    log.info('Validation plot saved → %s', val_png)
    ds_am.close()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — OUTPUT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def summarise_outputs():
    """Log a directory listing of OUTPUT_DIR with file sizes."""
    log.info('── Section 9: Output summary ──────────────────────────────────')
    log.info('All outputs in: %s', OUTPUT_DIR)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size_mb = os.path.getsize(fpath) / 1e6
        log.info('  %-50s  %8.1f MB', fname, size_mb)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='MORE dataset: IDF risk maps for Italy (1991–2020)'
    )
    p.add_argument('--data-root',  default=DATA_ROOT,
                   help='Root folder with more_YYYY/ sub-directories')
    p.add_argument('--output-dir', default=OUTPUT_DIR,
                   help='Folder for all output NetCDF, GeoTIFF and PNG files')
    p.add_argument('--years',      default=f'{YEARS[0]}-{YEARS[-1]}',
                   help='Year range, e.g. 1991-2020')
    p.add_argument('--skip-plots', action='store_true',
                   help='Skip all matplotlib output (useful for headless runs)')
    p.add_argument('--log-file',   default=LOG_FILE,
                   help='Optional path for a persistent log file')
    return p.parse_args()


def main():
    global DATA_ROOT, OUTPUT_DIR, YEARS, LOG_FILE

    args = _parse_args()
    DATA_ROOT  = args.data_root
    OUTPUT_DIR = args.output_dir
    LOG_FILE   = args.log_file

    # Parse year range
    start_yr, end_yr = map(int, args.years.split('-'))
    YEARS = list(range(start_yr, end_yr + 1))

    _setup_logging(LOG_FILE)

    log.info('════════════════════════════════════════════════════════════════')
    log.info('MORE IDF analysis — start')
    log.info('  DATA_ROOT  : %s', DATA_ROOT)
    log.info('  OUTPUT_DIR : %s', OUTPUT_DIR)
    log.info('  Years      : %d – %d  (%d years)', YEARS[0], YEARS[-1], len(YEARS))
    log.info('  Durations  : %s h', DURATIONS)
    log.info('  Return per.: %s yr', RETURN_PERIODS.tolist())
    log.info('  cartopy    : %s', HAS_CARTOPY)
    log.info('  rasterio   : %s', HAS_RASTERIO)
    log.info('════════════════════════════════════════════════════════════════')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Inspect dataset & auto-detect settings ─────────────────────────────
    test_file = os.path.join(
        DATA_ROOT, 'more_2020', '2020', 'moloch_tp_202005_zip_masked.nc'
    )
    tp_var, deacc = inspect_dataset(test_file)

    # ── 2. Annual-maximum stacks ──────────────────────────────────────────────
    build_annual_maxima(tp_var, deacc)

    # ── 3. GEV fitting & return-value maps ───────────────────────────────────
    compute_idf_return_values()

    # ── 4–6. Plots & validation ───────────────────────────────────────────────
    if not args.skip_plots:
        plot_idf_maps()
        plot_idf_curves()
        validate_gev()
    else:
        log.info('--skip-plots set: matplotlib output suppressed.')

    # ── 7. Summary ────────────────────────────────────────────────────────────
    summarise_outputs()

    log.info('MORE IDF analysis — complete.')


if __name__ == '__main__':
    main()
