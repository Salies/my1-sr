#!/usr/bin/env python3
"""
get_era5_blh.py
---------------
Download hourly ERA5 boundary layer height (BLH) for the grid cell containing
London Marylebone Road (MY1, an AURN air-quality site) over a date range, and
write a tidy hourly CSV.

Why ERA5 for BLH
----------------
Boundary layer height is not a routine surface observation -- there is no MIDAS
"nearest station" for it. ERA5 provides it as a modelled variable
('boundary_layer_height'), hourly, on a ~0.25 deg (~31 km) grid, 1940->~5 days
behind real time. That makes it the conventional open-access choice to pair with
air-quality monitoring. Caveat: it is a bulk-Richardson diagnostic at 31 km, so
it smooths over London's urban heat island -- fine as a covariate for MY1, but
it is a model estimate, not a measurement.

What this script does
---------------------
  1. Requests ERA5 single-levels 'boundary_layer_height' from the Copernicus
     Climate Data Store (CDS) for a small bounding box around MY1, hourly, over
     the requested date range.
  2. Downloads NetCDF, opens it, selects the grid cell nearest MY1.
  3. Writes an hourly CSV: time_utc, blh_m  (plus the cell's lat/lon in a header
     comment so you know which grid point you got).

Access
------
Free CDS account required. Configure credentials once, then run.
  1. Register: https://cds.climate.copernicus.eu/
  2. Accept the ERA5 licence (on the dataset's "Download data" tab, once).
  3. Put your API key in ~/.cdsapirc  (see https://cds.climate.copernicus.eu/how-to-api):
         url: https://cds.climate.copernicus.eu/api
         key: <YOUR-KEY>
     ...or pass --cds-url / --cds-key on the command line.

Dependencies
------------
  pip install "cdsapi>=0.7" xarray netCDF4 pandas numpy
  (netCDF4 is the NetCDF backend for xarray; numpy/pandas for tidy output.)

Usage
-----
  python get_era5_blh.py --start 2019-01-01 --end 2019-12-31
  python get_era5_blh.py --start 2018-01-01 --end 2024-12-31 --out my1_blh.csv
  # keep the raw NetCDF too:
  python get_era5_blh.py --start 2019-01-01 --end 2019-03-31 --keep-nc

Notes
-----
* ERA5 timestamps are UTC. MY1 / airqualityengland data are GMT hour-ending;
  ERA5 BLH is an instantaneous value at the stated hour. Align carefully when
  you merge -- see the note printed at the end of a run.
* Large multi-year requests are queued by the CDS and can take a while. The
  request is chunked by year so a long range doesn't hit per-request limits and
  so a failure only loses one year.
* BLH units are metres.
"""

import argparse
import calendar
import os
import sys
import tempfile
from datetime import date

# MY1 - London Marylebone Road
MY1_LAT = 51.522530
MY1_LON = -0.154611

DATASET = "reanalysis-era5-single-levels"
VARIABLE = "boundary_layer_height"

# A small bounding box around MY1. ERA5 is 0.25 deg; +/-0.125 guarantees we
# straddle the nearest grid point without pulling a large area. area = [N,W,S,E].
_HALF = 0.125
AREA = [MY1_LAT + _HALF, MY1_LON - _HALF, MY1_LAT - _HALF, MY1_LON + _HALF]

HOURS = [f"{h:02d}:00" for h in range(24)]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]


def daterange_years(start, end):
    """Yield (year, months, days) chunks, one per calendar year in [start,end].

    months/days are restricted so the first and last years don't request dates
    outside the range. Middle years get all months/all days.
    """
    y = start.year
    while y <= end.year:
        y_start = date(y, 1, 1)
        y_end = date(y, 12, 31)
        lo = max(start, y_start)
        hi = min(end, y_end)
        if lo.year == hi.year and lo.month == hi.month:
            months = [f"{lo.month:02d}"]
            days = [f"{d:02d}" for d in range(lo.day, hi.day + 1)]
        else:
            months = [f"{m:02d}" for m in range(lo.month, hi.month + 1)]
            # Request all valid days; CDS ignores days that don't exist in a
            # month, and out-of-range days at the very edges are trimmed after
            # download by the time filter. Simpler and robust.
            days = [f"{d:02d}" for d in range(1, 32)]
        yield y, months, days, lo, hi
        y += 1


def build_request(year, months, days):
    return {
        "product_type": ["reanalysis"],
        "variable": [VARIABLE],
        "year": [str(year)],
        "month": months,
        "day": days,
        "time": HOURS,
        "area": AREA,                 # [North, West, South, East]
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def download_year(client, year, months, days, workdir):
    target = os.path.join(workdir, f"era5_blh_{year}.nc")
    req = build_request(year, months, days)
    print(f"  requesting {year} ({months[0]}..{months[-1]}) -> CDS queue...",
          flush=True)
    client.retrieve(DATASET, req, target)
    return target


def open_and_extract(nc_paths, start, end):
    """Open the NetCDF file(s), pick the cell nearest MY1, return a tidy frame."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    ds = xr.open_mfdataset(nc_paths, combine="by_coords") if len(nc_paths) > 1 \
        else xr.open_dataset(nc_paths[0])

    # BLH variable name in the file is usually 'blh'; be tolerant.
    var = None
    for cand in ("blh", "boundary_layer_height"):
        if cand in ds.data_vars:
            var = cand
            break
    if var is None:
        # take the sole data variable if there's exactly one
        dvs = list(ds.data_vars)
        if len(dvs) == 1:
            var = dvs[0]
        else:
            raise SystemExit(f"Could not find BLH variable in NetCDF; "
                             f"data_vars = {dvs}")

    # Coordinate names differ across ERA5 downloads: latitude/longitude, and the
    # time axis may be 'time' or 'valid_time'.
    latname = "latitude" if "latitude" in ds.coords else "lat"
    lonname = "longitude" if "longitude" in ds.coords else "lon"
    timename = None
    for cand in ("valid_time", "time"):
        if cand in ds.coords or cand in ds.dims:
            timename = cand
            break
    if timename is None:
        raise SystemExit(f"Could not find a time coordinate; coords = "
                         f"{list(ds.coords)}")

    # ERA5 longitudes may be in the 0..360 domain. MY1 lon is negative, and a
    # small area box crossing the 0-deg meridian returns wrapped longitudes
    # (e.g. 359.75, 0.0, 0.25) which are NOT monotonic -- and .sel(nearest)
    # requires a monotonic index. So convert the FILE's longitudes to -180..180
    # (where MY1 lives naturally and the values stay monotonic), then select.
    lon_vals = ds[lonname].values
    if float(lon_vals.max()) > 180.0:
        ds = ds.assign_coords({lonname: (((ds[lonname] + 180) % 360) - 180)})
        ds = ds.sortby(lonname)

    cell = ds[var].sel({latname: MY1_LAT, lonname: MY1_LON}, method="nearest")
    got_lat = float(cell[latname].values)
    got_lon = float(cell[lonname].values)
    if got_lon > 180:
        got_lon -= 360

    s = cell.to_series().dropna()
    df = s.reset_index()
    df.columns = [c.lower() for c in df.columns]
    # standardise the time column name
    tcol = "valid_time" if "valid_time" in df.columns else (
        "time" if "time" in df.columns else df.columns[0])
    df = df.rename(columns={tcol: "time_utc", var: "blh_m"})
    df = df[["time_utc", "blh_m"]].copy()

    # trim to exact requested range (guards the all-days-in-edge-year request)
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)  # inclusive end day
    df = df[(df["time_utc"] >= lo) & (df["time_utc"] < hi)]
    df = df.sort_values("time_utc").reset_index(drop=True)

    return df, got_lat, got_lon


def parse_date(s):
    try:
        y, m, d = (int(x) for x in s.split("-"))
        return date(y, m, d)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"date must be YYYY-MM-DD, got {s!r}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=parse_date, required=True,
                    help="Start date (inclusive), YYYY-MM-DD.")
    ap.add_argument("--end", type=parse_date, required=True,
                    help="End date (inclusive), YYYY-MM-DD.")
    ap.add_argument("--out", default="my1_era5_blh.csv",
                    help="Output CSV path (default: my1_era5_blh.csv).")
    ap.add_argument("--cds-url", default=None,
                    help="Override CDS API url (else ~/.cdsapirc or CDSAPI_URL).")
    ap.add_argument("--cds-key", default=None,
                    help="Override CDS API key (else ~/.cdsapirc or CDSAPI_KEY).")
    ap.add_argument("--keep-nc", action="store_true",
                    help="Keep the downloaded NetCDF file(s) next to the CSV.")
    ap.add_argument("--workdir", default=None,
                    help="Directory for intermediate NetCDF (default: temp dir, "
                         "or the CSV's folder with --keep-nc).")
    args = ap.parse_args()

    if args.end < args.start:
        raise SystemExit("--end is before --start.")

    try:
        import cdsapi
    except ImportError:
        raise SystemExit("cdsapi not installed. Run:\n"
                         '  pip install "cdsapi>=0.7" xarray netCDF4 pandas numpy')

    # Build the client. Credentials come from ~/.cdsapirc unless overridden.
    client_kwargs = {}
    url = args.cds_url or os.environ.get("CDSAPI_URL")
    key = args.cds_key or os.environ.get("CDSAPI_KEY")
    if url and key:
        client_kwargs = {"url": url, "key": key}
    try:
        client = cdsapi.Client(**client_kwargs)
    except Exception as e:
        raise SystemExit(
            f"Could not initialise CDS client: {e}\n"
            "Set up ~/.cdsapirc (see https://cds.climate.copernicus.eu/how-to-api)\n"
            "  url: https://cds.climate.copernicus.eu/api\n"
            "  key: <YOUR-KEY>\n"
            "and accept the ERA5 licence on the dataset's Download tab once.")

    # Decide where NetCDF lands.
    if args.keep_nc:
        workdir = args.workdir or os.path.dirname(os.path.abspath(args.out)) or "."
        tmp = None
    else:
        tmp = tempfile.TemporaryDirectory()
        workdir = args.workdir or tmp.name
    os.makedirs(workdir, exist_ok=True)

    print(f"MY1: {MY1_LAT}, {MY1_LON}")
    print(f"ERA5 variable: {VARIABLE}  ({DATASET})")
    print(f"Range: {args.start} .. {args.end}  (hourly, UTC)")
    print(f"Bounding box [N,W,S,E]: {AREA}\n")

    nc_paths = []
    try:
        for year, months, days, lo, hi in daterange_years(args.start, args.end):
            path = download_year(client, year, months, days, workdir)
            nc_paths.append(path)
    except Exception as e:
        raise SystemExit(f"CDS download failed: {e}\n"
                         "If this is a licence error, accept the ERA5 licence "
                         "once on the dataset's Download tab in the CDS web UI.")

    print("\nOpening NetCDF and extracting the cell nearest MY1...")
    df, got_lat, got_lon = open_and_extract(nc_paths, args.start, args.end)

    # Write CSV with a provenance header.
    with open(args.out, "w", newline="") as fh:
        fh.write(f"# ERA5 {VARIABLE} (m), hourly UTC\n")
        fh.write(f"# requested point: MY1 {MY1_LAT},{MY1_LON}\n")
        fh.write(f"# nearest ERA5 grid cell: {got_lat:.4f},{got_lon:.4f}\n")
        fh.write(f"# range: {args.start}..{args.end}  rows: {len(df)}\n")
        df.to_csv(fh, index=False)

    print(f"\nWrote {len(df)} hourly rows -> {args.out}")
    print(f"Nearest ERA5 grid cell: {got_lat:.4f}, {got_lon:.4f} "
          f"(~{_dist(got_lat, got_lon):.1f} km from MY1)")
    if args.keep_nc:
        print("Kept NetCDF: " + ", ".join(nc_paths))
    print("\nTIME ALIGNMENT NOTE:")
    print("  ERA5 BLH is an instantaneous value at each UTC hour.")
    print("  airqualityengland/MIDAS data are GMT hour-ending. GMT==UTC, but the")
    print("  hour-ending convention means their 13:00 covers 12:00-13:00, while")
    print("  ERA5 13:00 is the instant. Decide on your merge convention (e.g.")
    print("  shift, or accept the ~1h instantaneous-vs-mean difference).")

    if tmp is not None:
        tmp.cleanup()


def _dist(lat, lon):
    import math
    r = 6371.0088
    p1, p2 = math.radians(MY1_LAT), math.radians(lat)
    dphi = math.radians(lat - MY1_LAT)
    dlmb = math.radians(lon - MY1_LON)
    a = (math.sin(dphi/2)**2
         + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2)
    return 2 * r * math.asin(math.sqrt(a))


if __name__ == "__main__":
    main()
