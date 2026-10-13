
#!/usr/bin/env python3
"""Aggregate forest-age raster pixels into global 0.5-degree age distributions.

The age raster is assumed to represent forests in REFERENCE_YEAR (2020).
A START_YEAR of 1700 requires distinguishing age cohorts up to 320 years.
The last age class is open ended: age_320_330 represents all ages >= 320.

Only pixels with valid, nonnegative age values enter the distributions.
Probabilities are CONDITIONAL ON valid forest-age pixels, not forest cover
fractions relative to the full land or grid-cell area.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

# ============================================================
# 1. Default input/output paths and historical settings
# ============================================================
INPUT_TIF = Path(
    "/data/global_mapping/models_V100/"
    "fina_proditionA/Masked/AGE_MEAN_300.tif"
)
OUTPUT_TXT = Path("/data/global_carbon_sink/age_groups_0p5deg_1700.txt")

START_YEAR = 1700
REFERENCE_YEAR = 2020
AGE_STEP = 10
GRID_RES = 0.5
BLOCK_SIZE = 2048

LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_TIF)
    parser.add_argument("--output", type=Path, default=OUTPUT_TXT)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--reference-year", type=int, default=REFERENCE_YEAR)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input raster not found: {args.input}")
    if args.reference_year <= args.start_year:
        raise ValueError("Reference year must be later than start year.")
    if args.block_size <= 0:
        raise ValueError("Block size must be positive.")

    # ========================================================
    # 2. Derive age groups from the historical start year
    # ========================================================
    lookback_years = args.reference_year - args.start_year
    oldest_group_start = (lookback_years // AGE_STEP) * AGE_STEP
    n_age_groups = oldest_group_start // AGE_STEP + 1
    age_edges = np.arange(n_age_groups + 1, dtype=np.int64) * AGE_STEP
    age_columns = [
        f"age_{age_edges[i]}_{age_edges[i + 1]}"
        for i in range(n_age_groups)
    ]
    # Example: 2020 - 1700 = 320 years, giving 33 groups.
    # The final 'age_320_330' group includes ALL ages >= 320.

    n_lon = int(round((LON_MAX - LON_MIN) / GRID_RES))
    n_lat = int(round((LAT_MAX - LAT_MIN) / GRID_RES))
    n_cells = n_lon * n_lat

    age_counts = np.zeros(n_cells * n_age_groups, dtype=np.uint64)
    valid_age_pixels = np.zeros(n_cells, dtype=np.uint64)

    total_valid_pixels = 0
    total_negative_pixels = 0
    observed_min_age = np.inf
    observed_max_age = -np.inf

    # ========================================================
    # 3. Process the raster block by block
    # ========================================================
    with rasterio.open(args.input) as src:
        if src.crs is None:
            raise RuntimeError("Input raster has no CRS.")

        transformer = None
        if src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(
                src.crs, "EPSG:4326", always_xy=True
            )

        print("=" * 72)
        print("GLOBAL FOREST-AGE DISTRIBUTION")
        print("=" * 72)
        print(f"Input: {args.input}")
        print(f"Raster CRS: {src.crs}")
        print(f"Raster dimensions: {src.width} x {src.height}")
        print(f"NoData: {src.nodata}")
        print(f"Historical start year: {args.start_year}")
        print(f"Age reference year: {args.reference_year}")
        print(f"Lookback period: {lookback_years} years")
        print(f"Age-group interval: {AGE_STEP} years")
        print(f"Number of age groups: {n_age_groups}")
        print(f"Oldest group: {age_columns[-1]} (ages >= {oldest_group_start})")

        n_block_rows = (src.height + args.block_size - 1) // args.block_size
        n_block_cols = (src.width + args.block_size - 1) // args.block_size
        total_blocks = n_block_rows * n_block_cols
        block_number = 0
        transform = src.transform

        for row_off in range(0, src.height, args.block_size):
            height = min(args.block_size, src.height - row_off)

            for col_off in range(0, src.width, args.block_size):
                width = min(args.block_size, src.width - col_off)
                block_number += 1
                window = Window(col_off, row_off, width, height)
                age_block = src.read(1, window=window, masked=True)

                data = np.asarray(age_block.data, dtype=np.float64)
                valid = (~np.ma.getmaskarray(age_block)) & np.isfinite(data)
                if not np.any(valid):
                    continue

                local_rows, local_cols = np.nonzero(valid)
                ages = data[local_rows, local_cols]
                nonnegative = ages >= 0.0
                total_negative_pixels += int(np.count_nonzero(~nonnegative))

                local_rows = local_rows[nonnegative]
                local_cols = local_cols[nonnegative]
                ages = ages[nonnegative]
                if ages.size == 0:
                    continue

                # Compute original raster pixel-center coordinates.
                global_rows = row_off + local_rows.astype(np.float64)
                global_cols = col_off + local_cols.astype(np.float64)
                x = (
                    transform.a * (global_cols + 0.5)
                    + transform.b * (global_rows + 0.5)
                    + transform.c
                )
                y = (
                    transform.d * (global_cols + 0.5)
                    + transform.e * (global_rows + 0.5)
                    + transform.f
                )

                if transformer is None:
                    lon, lat = x, y
                else:
                    lon, lat = transformer.transform(x, y)
                    lon = np.asarray(lon, dtype=np.float64)
                    lat = np.asarray(lat, dtype=np.float64)

                # Filter invalid coordinates before integer conversion.
                coord_ok = np.isfinite(lon) & np.isfinite(lat)
                if not np.any(coord_ok):
                    continue
                lon = lon[coord_ok]
                lat = lat[coord_ok]
                ages = ages[coord_ok]

                # Assign pixel centers to LPJ-GUESS 0.5-degree cells.
                lon = ((lon + 180.0) % 360.0) - 180.0
                lon_index = np.floor((lon - LON_MIN) / GRID_RES).astype(np.int64)
                lat_index = np.floor((lat - LAT_MIN) / GRID_RES).astype(np.int64)
                spatial_ok = (
                    (lon_index >= 0)
                    & (lon_index < n_lon)
                    & (lat_index >= 0)
                    & (lat_index < n_lat)
                )
                if not np.any(spatial_ok):
                    continue

                lon_index = lon_index[spatial_ok]
                lat_index = lat_index[spatial_ok]
                ages = ages[spatial_ok]

                total_valid_pixels += ages.size
                observed_min_age = min(observed_min_age, float(ages.min()))
                observed_max_age = max(observed_max_age, float(ages.max()))

                cell_index = lat_index * n_lon + lon_index
                age_group = np.floor(ages / AGE_STEP).astype(np.int64)
                age_group = np.clip(age_group, 0, n_age_groups - 1)

                unique_cells, counts = np.unique(cell_index, return_counts=True)
                valid_age_pixels[unique_cells] += counts.astype(np.uint64)

                combined_index = cell_index * n_age_groups + age_group
                unique_combined, counts = np.unique(
                    combined_index, return_counts=True
                )
                age_counts[unique_combined] += counts.astype(np.uint64)

                if block_number % 20 == 0 or block_number == total_blocks:
                    print(f"Processed blocks: {block_number}/{total_blocks}")

    # ========================================================
    # 4. Calculate probability distributions for forest cells
    # ========================================================
    age_counts = age_counts.reshape(n_cells, n_age_groups)
    forest_cells = np.flatnonzero(valid_age_pixels > 0)
    if forest_cells.size == 0:
        raise RuntimeError("No valid forest-age pixels were found.")

    selected_counts = age_counts[forest_cells]
    selected_totals = valid_age_pixels[forest_cells]
    if not np.array_equal(selected_counts.sum(axis=1), selected_totals):
        raise RuntimeError("Age-group pixel-count QC failed.")

    probabilities = selected_counts / selected_totals[:, None]
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1e-10, rtol=0):
        raise RuntimeError("Age-group probabilities do not sum to 1.")

    lat_index = forest_cells // n_lon
    lon_index = forest_cells % n_lon
    lon_center = LON_MIN + (lon_index + 0.5) * GRID_RES
    lat_center = LAT_MIN + (lat_index + 0.5) * GRID_RES

    if not (
        np.allclose((lon_center * 2.0) % 1.0, 0.5, atol=1e-10)
        and np.allclose((lat_center * 2.0) % 1.0, 0.5, atol=1e-10)
    ):
        raise RuntimeError("Invalid 0.5-degree grid centers detected.")

    # ========================================================
    # 5. Build and export the output table
    # ========================================================
    output = pd.DataFrame(probabilities, columns=age_columns)
    output.insert(0, "lat", lat_center)
    output.insert(0, "lon", lon_center)
    output["forest_pixel_count"] = selected_totals
    output["probability_sum"] = probability_sums
    output = output.sort_values(
        ["lat", "lon"], ascending=[False, True]
    ).reset_index(drop=True)
    output.insert(0, "ID", np.arange(1, len(output) + 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False, float_format="%.12f")

    print("\n" + "=" * 72)
    print("FINISHED")
    print("=" * 72)
    print(f"Output: {args.output}")
    print(f"0.5-degree forest cells: {len(output)}")
    print(f"Valid forest-age pixels: {total_valid_pixels:,}")
    print(f"Excluded negative-age pixels: {total_negative_pixels:,}")
    print(f"Observed age range: {observed_min_age:.2f} to {observed_max_age:.2f}")
    print(f"Final age class: {age_columns[-1]} represents >= {oldest_group_start}")
    print(f"Pixels in final age class: {selected_counts[:, -1].sum():,}")
    print("Note: probabilities describe age composition among valid forest pixels.")
    print("Note: this script does NOT reconstruct annual forest cover.")
    print("\nFirst five output rows:")
    print(output.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
