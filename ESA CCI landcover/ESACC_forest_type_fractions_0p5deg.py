#!/usr/bin/env python3
"""Count ESA forest types directly from a raster on the age-table grid.

Output columns: lon, lat, ND, BD, MX, UNK.
Known cells: normalize ND/BD/MX pixel counts; UNK is zero.
Cells without target pixels: ND=BD=MX=0 and UNK=1.
Use native pixel centers, without resampling or area weighting.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window

BASE = Path("/data/global_carbon_sink")
DEFAULT_RASTER = BASE / (
    "ESA_land_cover/ESA-CCI-LULC-300m-P1Y-2020-v2.1.1.tif"
)
DEFAULT_REFERENCE = BASE / "age_groups_0p5deg.txt"
DEFAULT_OUTPUT = BASE / (
    "ESA_land_cover/forest_type_fractions_0p5deg_2020.txt"
)
TYPE_CODES = {
    "ND": (70, 71, 72, 80, 81, 82),
    "BD": (50, 60, 61, 62),
    "MX": (90,),
}
TYPE_NAMES = ["ND", "BD", "MX", "UNK"]
NROWS, NCOLS = 360, 720
GRID_SIZE = 0.5
COORD_TOL = 1e-8
SCALE = 10 ** 10

# Buckets: 0=invalid, 1=ND, 2=BD, 3=MX, 4=other valid classes.
NBUCKETS = 5
LOOKUP = np.full(256, 4, dtype=np.uint8)
LOOKUP[0] = 0
for bucket_id, codes in enumerate(TYPE_CODES.values(), start=1):
    LOOKUP[list(codes)] = bucket_id


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster", type=Path, default=DEFAULT_RASTER)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-qc", action="store_true",
        help="Also write pixel-count QC and unknown-cell tables."
    )
    return parser.parse_args()


def load_reference(path):
    """Preserve the reference coordinates and row order without snapping."""
    if not path.is_file():
        raise FileNotFoundError(f"Reference table not found: {path}")
    header = pd.read_csv(path, sep=r"\s+", nrows=0, encoding="utf-8-sig")
    lookup = {str(c).strip().lower(): c for c in header.columns}
    if len(lookup) != len(header.columns):
        raise ValueError("Duplicate reference column names.")
    if not {"lon", "lat"}.issubset(lookup):
        raise ValueError("The reference table must contain lon and lat.")
    mapping = {lookup["lon"]: "lon", lookup["lat"]: "lat"}
    frame = pd.read_csv(
        path, sep=r"\s+", usecols=list(mapping), dtype=np.float64,
        encoding="utf-8-sig"
    ).rename(columns=mapping)[["lon", "lat"]]
    xy = frame.to_numpy()
    if frame.empty or not np.isfinite(xy).all():
        raise ValueError("Reference coordinates are empty or non-finite.")
    lon, lat = xy[:, 0], xy[:, 1]
    if np.any((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)):
        raise ValueError("Coordinates outside geographic bounds.")
    cols = np.rint((lon + 179.75) / GRID_SIZE).astype(np.int64)
    rows = np.rint((89.75 - lat) / GRID_SIZE).astype(np.int64)
    valid = (
        (rows >= 0) & (rows < NROWS) & (cols >= 0) & (cols < NCOLS)
        & (np.abs(lon - (-179.75 + cols * GRID_SIZE)) <= COORD_TOL)
        & (np.abs(lat - (89.75 - rows * GRID_SIZE)) <= COORD_TOL)
    )
    if not valid.all():
        raise ValueError(
            "Expected 0.5-degree grid centers ending in .25 or .75.\n"
            + frame.loc[~valid].head().to_string(index=False)
        )
    if np.unique(rows * NCOLS + cols).size != len(frame):
        raise ValueError("Duplicate reference grid cells.")
    return frame, rows, cols


def stable_floor(values):
    nearest = np.rint(values)
    return np.floor(
        np.where(np.abs(values - nearest) <= 1e-10, nearest, values)
    ).astype(np.int64)


def validate_raster(src):
    if src.count != 1:
        raise ValueError("Expected a single-band categorical raster.")
    if src.crs is None or src.crs != CRS.from_epsg(4326):
        raise ValueError(f"Expected EPSG:4326; found {src.crs}. No reprojection is done.")
    t = src.transform
    if not np.isfinite(np.asarray(tuple(t), dtype=float)).all():
        raise ValueError("Invalid raster transform.")
    if abs(t.b) > 1e-12 or abs(t.d) > 1e-12 or t.a <= 0 or t.e >= 0:
        raise ValueError("Expected an unrotated, north-up raster.")
    if np.dtype(src.dtypes[0]).kind not in "iu":
        raise ValueError("Expected integer land-cover codes.")
    b = src.bounds
    if (b.left < -180 - COORD_TOL or b.right > 180 + COORD_TOL
            or b.bottom < -90 - COORD_TOL or b.top > 90 + COORD_TOL):
        raise ValueError("Raster bounds must be inside [-180,180] x [-90,90].")


def window_shape(src):
    """Group native blocks into bounded processing windows."""
    bh, bw = src.block_shapes[0]
    budget = 2_097_152
    if bh * bw > budget:
        print("WARNING: Large native blocks may increase GDAL memory use.", flush=True)
        return min(src.height, 1024), min(src.width, 1024)
    width = min(src.width, max(bw, (1024 // bw) * bw))
    height = min(src.height, max(bh, (budget // width // bh) * bh))
    return height, width


def count_pixels(path, ref_rows, ref_cols):
    wanted = np.zeros((NROWS, NCOLS), dtype=bool)
    wanted[ref_rows, ref_cols] = True
    counts = np.zeros((NROWS, NCOLS, NBUCKETS), dtype=np.int64)
    with rasterio.open(path) as src:
        validate_raster(src)
        t = src.transform
        x = t.c + (np.arange(src.width, dtype=np.float64) + 0.5) * t.a
        y = t.f + (np.arange(src.height, dtype=np.float64) + 0.5) * t.e
        # Boundaries: west <= x < east and south < y <= north.
        col_ids = stable_floor((x + 180.0) / GRID_SIZE)
        row_ids = stable_floor((90.0 - y) / GRID_SIZE)
        if (np.any((col_ids < 0) | (col_ids >= NCOLS))
                or np.any((row_ids < 0) | (row_ids >= NROWS))):
            raise ValueError("A raster pixel center is outside the global grid.")
        win_h, win_w = window_shape(src)
        total = math.ceil(src.height / win_h) * math.ceil(src.width / win_w)
        print(f"Raster: {path}")
        print(f"CRS: {src.crs}; size: {src.width:,} x {src.height:,}")
        print(f"Pixel size: {src.res}; NoData: {src.nodata}")
        print(f"Native blocks: {src.block_shapes[0]}; window: {win_h} x {win_w}")
        print(f"Windows to examine: {total:,}", flush=True)
        examined = read_windows = 0
        last_report = time.monotonic()
        for r0 in range(0, src.height, win_h):
            height = min(win_h, src.height - r0)
            rr = row_ids[r0:r0 + height]
            rmin, rmax = int(rr[0]), int(rr[-1])
            for c0 in range(0, src.width, win_w):
                examined += 1
                width = min(win_w, src.width - c0)
                cc = col_ids[c0:c0 + width]
                cmin, cmax = int(cc[0]), int(cc[-1])
                if wanted[rmin:rmax + 1, cmin:cmax + 1].any():
                    array = src.read(
                        1, window=Window(c0, r0, width, height), masked=True
                    )
                    data = array.data
                    if data.dtype == np.dtype("uint8"):
                        bucket = LOOKUP[data]
                    else:
                        bucket = np.full(data.shape, 4, dtype=np.uint8)
                        in_lookup = (data >= 0) & (data < 256)
                        bucket[in_lookup] = LOOKUP[data[in_lookup].astype(np.uint8)]
                    bucket[np.ma.getmaskarray(array)] = 0
                    nr, nc = rmax - rmin + 1, cmax - cmin + 1
                    keys = (rr - rmin)[:, None] * nc + (cc - cmin)[None, :]
                    keys *= NBUCKETS
                    keys += bucket
                    hist = np.bincount(
                        keys.ravel(), minlength=nr * nc * NBUCKETS
                    ).reshape(nr, nc, NBUCKETS)
                    counts[rmin:rmax + 1, cmin:cmax + 1] += hist
                    read_windows += 1
                    del array, data, bucket, keys, hist
                now = time.monotonic()
                if now - last_report >= 5 or examined == total:
                    print(
                        f"Examined {examined:,}/{total:,} windows "
                        f"({100 * examined / total:.1f}%); read {read_windows:,}",
                        flush=True
                    )
                    last_report = now
        bounds = src.bounds
    return counts[ref_rows, ref_cols], bounds


def fraction_units(target_counts):
    """Return integer units summing to 10**10, including unknown cells."""
    totals = target_counts.sum(axis=1)
    known = totals > 0
    units = np.zeros((len(totals), 4), dtype=np.int64)
    units[~known, 3] = SCALE
    if known.any():
        scaled = target_counts[known] / totals[known, None] * SCALE
        parts = np.floor(scaled).astype(np.int64)
        remaining = SCALE - parts.sum(axis=1)
        if np.any((remaining < 0) | (remaining > 3)):
            raise RuntimeError("Unexpected rounding remainder.")
        order = np.argsort(-(scaled - parts), axis=1, kind="stable")
        for k in range(3):
            selected = np.flatnonzero(remaining > k)
            parts[selected, order[selected, k]] += 1
        units[known, :3] = parts
    if np.any(units < 0) or not np.all(units.sum(axis=1) == SCALE):
        raise RuntimeError("Fraction sum-to-one check failed.")
    return units, totals, known


def write_table(frame, path, expected_units=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        frame.to_csv(temp, sep="\t", index=False, float_format="%.10f")
        if expected_units is not None:
            check = pd.read_csv(temp, sep=r"\s+")
            if list(check.columns) != ["lon", "lat"] + TYPE_NAMES or len(check) != len(frame):
                raise RuntimeError("Output columns or row count differ.")
            if not np.allclose(
                check[["lon", "lat"]], frame[["lon", "lat"]], atol=1e-10, rtol=0
            ):
                raise RuntimeError("Output coordinates or row order differ.")
            values = check[TYPE_NAMES].to_numpy()
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise RuntimeError("Invalid output fractions.")
            actual_units = np.rint(values * SCALE).astype(np.int64)
            if not np.array_equal(actual_units, expected_units):
                raise RuntimeError("Exported decimal precision check failed.")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main():
    args = parse_args()
    if not args.raster.is_file():
        raise FileNotFoundError(f"Raster not found: {args.raster}")
    qc_path = args.output.with_name(args.output.stem + "_QC.txt")
    unk_path = args.output.with_name(args.output.stem + "_UNK_cells.txt")
    inputs = {args.raster.resolve(), args.reference.resolve()}
    destinations = [args.output] + ([qc_path, unk_path] if args.write_qc else [])
    if any(path.resolve() in inputs for path in destinations):
        raise ValueError("An output path would overwrite an input.")
    reference, rows, cols = load_reference(args.reference)
    print(f"Reference: {args.reference}; cells: {len(reference):,}")
    print(f"Target codes: {TYPE_CODES}")
    print("No target pixels: ND=BD=MX=0, UNK=1.", flush=True)
    counts, bounds = count_pixels(args.raster, rows, cols)
    units, target_total, known = fraction_units(counts[:, 1:4])
    output = reference.copy()
    output[TYPE_NAMES] = units / SCALE

    all_pixels = counts.sum(axis=1)
    valid_pixels = all_pixels - counts[:, 0]
    lon, lat = reference["lon"].to_numpy(), reference["lat"].to_numpy()
    full_extent = (
        (lon - 0.25 >= bounds.left - COORD_TOL)
        & (lon + 0.25 <= bounds.right + COORD_TOL)
        & (lat - 0.25 >= bounds.bottom - COORD_TOL)
        & (lat + 0.25 <= bounds.top + COORD_TOL)
    )
    status = np.full(len(reference), "OK", dtype=object)
    status[known & ~full_extent] = "PARTIAL_RASTER_EXTENT"
    status[~known] = "NO_TARGET_FOREST_PIXELS"
    status[valid_pixels == 0] = "NO_VALID_PIXELS"
    status[all_pixels == 0] = "NO_RASTER_PIXELS"
    if args.write_qc:
        qc = output.copy()
        qc[["ND_pixels", "BD_pixels", "MX_pixels"]] = counts[:, 1:4]
        qc["target_forest_pixels"] = target_total
        qc["other_valid_pixels"] = counts[:, 4]
        qc["nodata_pixels"] = counts[:, 0]
        qc["in_raster_pixels"] = all_pixels
        qc["full_raster_extent"] = full_extent
        qc["status"] = status
        write_table(qc, qc_path)
        write_table(qc.loc[~known], unk_path)
        print(f"QC table: {qc_path}")
        print(f"Unknown-cell table: {unk_path}")
    write_table(output, args.output, expected_units=units)
    print(f"Known type cells: {int(known.sum()):,}")
    print(f"UNK=1 cells: {int((~known).sum()):,}")
    print(pd.Series(status).value_counts().to_string())
    if (~full_extent).any():
        print("WARNING: Some cells lack full raster coverage; review --write-qc outputs.")
    print(f"SUCCESS: {args.output}")
    print("PASS: coordinates and row order match the age table.")
    print("PASS: ND + BD + MX + UNK = 1 at 10 decimal places; no NaN values.")
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)