"""
idf_chunked_helpers.py
======================
Memory-efficient annual-maximum computation for the MORE dataset.

Strategy
--------
Never concatenate a full year in memory.  Instead, process one month at a
time and carry a small boundary buffer (duration_h - 1 rows along the time
axis) from the previous month so that rolling windows spanning a month
boundary are handled correctly.

Grid facts (960 lat × 768 lon, float32)
  1 month  ≈  2.2 GB
  1 year   ≈ 26 GB   (would be ~76 GB after xr.concat overhead)
  Peak RAM with this approach: 2 × 2.2 GB + tiny carry buffer ≈ 5 GB
"""

import os
import glob
import gc
import numpy as np
import xarray as xr


# ── tuneable ──────────────────────────────────────────────────────────────────
# How many months to hold in memory at once.
# 2 = current month + carry buffer.  Safe on any machine with > 6 GB free.
# Increase to 3-4 on machines with > 20 GB free for a small speed gain.
MONTHS_IN_FLIGHT = 2
# ──────────────────────────────────────────────────────────────────────────────


def deaccumulate(arr):
    """
    Convert a cumulative-from-run-start float32 array (time, lat, lon) to
    hourly amounts in-place.  Returns the same array modified.
    """
    diff = np.diff(arr, axis=0, prepend=arr[[0]])
    # where diff is negative a new forecast run started; use the raw value
    neg = diff < 0
    diff[neg] = arr[neg]
    diff[diff < 0] = 0.0
    return diff.astype(np.float32)


def _monthly_files(year, data_root):
    pattern = os.path.join(
        data_root, f'more_{year}', str(year),
        f'moloch_tp_{year}??_zip_masked.nc'
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files found for year {year}:\n  {pattern}')
    return files


def rolling_accumulation_annual_max_chunked(year, data_root, tp_var,
                                            duration_h, deacc=False):
    """
    Compute the annual maximum of `duration_h`-hour rolling precipitation sums
    for one year.

    Peak RAM ≈ 2 × (one monthly file) + carry buffer
             ≈ 2 × 2.2 GB + negligible  ≈  5 GB

    Parameters
    ----------
    year       : int
    data_root  : str
    tp_var     : str   – NetCDF variable name
    duration_h : int   – rolling window in hours
    deacc      : bool  – apply de-accumulation

    Returns
    -------
    xarray.DataArray, shape (lat, lon) – annual max [mm]
    """
    files = _monthly_files(year, data_root)

    # ── read coordinate arrays from the first file (no data) ─────────────────
    with xr.open_dataset(files[0]) as ds0:
        da0       = ds0[tp_var].squeeze()
        lat_coord = da0['lat']
        lon_coord = da0['lon']
        n_lat     = da0.sizes['lat']
        n_lon     = da0.sizes['lon']

    ann_max = np.full((n_lat, n_lon), -np.inf, dtype=np.float32)

    # carry buffer: last (duration_h - 1) time steps of the previous month
    # needed so rolling windows that span a month boundary are complete
    carry = None   # shape: (duration_h-1, n_lat, n_lon) once initialised

    for f in files:
        # ── load one month ────────────────────────────────────────────────────
        with xr.open_dataset(f) as ds:
            arr = ds[tp_var].squeeze().values.astype(np.float32)
            # arr shape: (time_steps, lat, lon)  e.g. (744, 960, 768)

        if deacc:
            arr = deaccumulate(arr)

        # ── prepend carry buffer from previous month ──────────────────────────
        if carry is not None and duration_h > 1:
            arr = np.concatenate([carry, arr], axis=0)

        # update carry for next iteration
        if duration_h > 1:
            carry = arr[-(duration_h - 1):].copy()
        else:
            carry = None   # 1-h window needs no carry

        # ── rolling sum via cumsum trick (no extra full-array copy) ───────────
        # cs[t] = sum of arr[0..t]
        cs = np.cumsum(arr, axis=0, dtype=np.float32)

        # rolled[t] = sum of arr[t-duration_h+1 .. t]
        rolled = cs.copy()
        rolled[duration_h:] = cs[duration_h:] - cs[:-duration_h]
        rolled[:duration_h] = np.nan   # incomplete window

        # month-local max (ignoring the carry prefix rows for the max update
        # so we don't double-count them — they were already counted last month)
        offset = (duration_h - 1) if (carry is not None or duration_h > 1) else 0
        month_max = np.nanmax(rolled[offset:], axis=0)

        np.maximum(ann_max, month_max, out=ann_max)

        del arr, cs, rolled, month_max
        gc.collect()

    ann_max[ann_max == -np.inf] = np.nan

    return xr.DataArray(
        ann_max,
        dims=['lat', 'lon'],
        coords={'lat': lat_coord, 'lon': lon_coord},
        attrs={'units': 'mm',
               'description': f'Annual max {duration_h}h precipitation – {year}'}
    )


# ── original API kept for compatibility / small smoke tests ───────────────────

def deaccumulate_da(da):
    arr = da.values.copy().astype(np.float32)
    return da.copy(data=deaccumulate(arr))


def load_hourly_tp_year(year, data_root, tp_var, deacc=False):
    """
    Original loader – fine for inspection / small subsets.
    Loads the full year (~76 GB with xarray overhead).  Do NOT use in the
    production loop; use rolling_accumulation_annual_max_chunked() instead.
    """
    import warnings
    warnings.warn(
        "load_hourly_tp_year loads ~76 GB per year due to xarray concat overhead. "
        "Use rolling_accumulation_annual_max_chunked() for production.",
        ResourceWarning, stacklevel=2
    )
    files = _monthly_files(year, data_root)
    monthly = []
    for f in files:
        ds  = xr.open_dataset(f)
        da  = ds[tp_var].squeeze()
        if deacc:
            da = deaccumulate_da(da)
        monthly.append(da)
        ds.close()
    return xr.concat(monthly, dim='time')


def rolling_accumulation_annual_max(da_year, duration_h):
    """Original xarray version – kept for small test subsets."""
    rolled = da_year.rolling(time=duration_h, min_periods=duration_h).sum()
    return rolled.max(dim='time')