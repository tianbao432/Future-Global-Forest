#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate Besnard forest ages to fixed 15-class, 0.5-degree distributions.

Classes: [0,10), [10,20), ..., [130,140), [140,infinity).
For compatibility with existing data tables, the last open-ended class is
named 'age_140_150'. Its actual meaning is age >= 140 years.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


# ============================================================
# 1. Configuration
# ============================================================
INPUT_TIF = Path(
    "/data/global_carbon_sink/Besnard_map/ForestAge_TC000_Masked.tif"
)
OUTPUT_TXT = Path(
    "/data/global_carbon_sink/Besnard_age_groups_0p5deg_1700.txt"
)

GRID_RES = 0.5
AGE_STEP = 10
N_AGE_GROUPS = 15
BLOCK_SIZE = 1024

LON_MIN, LON_MAX = -180.0, 180.0
LAT_MIN, LAT_MAX = -90.0, 90.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_TIF)
    parser.add_argument("--output", type=Path, default=OUTPUT_TXT)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input raster not found: {args.input}")
    if args.block_size < 1:
        raise ValueError("Block size must be positive.")

    # ========================================================
    # 2. Fix the age grouping to exactly 15 classes
    # ========================================================
    age_columns = [
        f"age_{i * AGE_STEP}_{(i + 1) * AGE_STEP}"
        for i in range(N_AGE_GROUPS)
    ]
    last_group_start = (N_AGE_GROUPS - 1) * AGE_STEP
    last_group_upper_label = N_AGE_GROUPS * AGE_STEP

    n_lon = int(round((LON_MAX - LON_MIN) / GRID_RES))
    n_lat = int(round((LAT_MAX - LAT_MIN) / GRID_RES))
    n_cells = n_lon * n_lat

    age_counts = np.zeros(n_cells * N_AGE_GROUPS, dtype=np.int64)
    pixel_counts = np.zeros(n_cells, dtype=np.int64)

    total_valid = 0
    negative_count = 0
    invalid_coordinate_count = 0
    outside_grid_count = 0
    older_than_150_count = 0
    observed_min_age = np.inf
    observed_max_age = -np.inf

    print("=" * 70)
    print("BESNARD FOREST AGE - FIXED 15-CLASS DISTRIBUTIONS")
    print("=" * 70)
    print("Input:", args.input)
    print("Output:", args.output)
    print("Age columns:", ", ".join(age_columns))
    print(f"Final column {age_columns[-1]} includes all ages >= 140.")

    # ========================================================
    # 3. Process the original raster block by block
    # ========================================================
    with rasterio.open(args.input) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS.")

        transformer = None
        if src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(
                src.crs, "EPSG:4326", always_xy=True
            )

        print("Raster CRS:", src.crs)
        print("Raster shape:", (src.height, src.width))
        print("Raster NoData:", src.nodata)

        n_block_rows = (src.height + args.block_size - 1) // args.block_size
        n_block_cols = (src.width + args.block_size - 1) // args.block_size
        total_blocks = n_block_rows * n_block_cols
        block_number = 0
        transform = src.transform

        for row_off in range(0, src.height, args.block_size):
            h = min(args.block_size, src.height - row_off)
            for col_off in range(0, src.width, args.block_size):
                w = min(args.block_size, src.width - col_off)
                block_number += 1
                window = Window(col_off, row_off, w, h)

                age_block = src.read(1, window=window, masked=True)
                data = age_block.data
                valid = (~np.ma.getmaskarray(age_block)) & np.isfinite(data)

                if not np.any(valid):
                    if block_number % 20 == 0 or block_number == total_blocks:
                        print(f"Processed {block_number}/{total_blocks} blocks", flush=True)
                    continue

                local_rows, local_cols = np.nonzero(valid)
                ages = np.asarray(data[local_rows, local_cols], dtype=np.float64)

                nonnegative = ages >= 0.0
                negative_count += int(np.count_nonzero(~nonnegative))
                local_rows = local_rows[nonnegative]
                local_cols = local_cols[nonnegative]
                ages = ages[nonnegative]
                if ages.size == 0:
                    continue

                # Pixel centers in the original raster coordinate system.
                rows = row_off + local_rows
                cols = col_off + local_cols
                x = (
                    transform.a * (cols + 0.5)
                    + transform.b * (rows + 0.5)
                    + transform.c
                )
                y = (
                    transform.d * (cols + 0.5)
                    + transform.e * (rows + 0.5)
                    + transform.f
                )

                if transformer is not None:
                    x, y = transformer.transform(x, y)

                lon = np.asarray(x, dtype=np.float64)
                lat = np.asarray(y, dtype=np.float64)
                coordinate_ok = np.isfinite(lon) & np.isfinite(lat)
                invalid_coordinate_count += int(np.count_nonzero(~coordinate_ok))
                lon = lon[coordinate_ok]
                lat = lat[coordinate_ok]
                ages = ages[coordinate_ok]
                if ages.size == 0:
                    continue

                # Assign pixel centers to global 0.5-degree cells.
                lon = ((lon + 180.0) % 360.0) - 180.0
                lon_idx = np.floor((lon - LON_MIN) / GRID_RES).astype(np.int64)
                lat_idx = np.floor((lat - LAT_MIN) / GRID_RES).astype(np.int64)
                spatial_ok = (
                    (lon_idx >= 0) & (lon_idx < n_lon)
                    & (lat_idx >= 0) & (lat_idx < n_lat)
                )
                outside_grid_count += int(np.count_nonzero(~spatial_ok))
                lon_idx = lon_idx[spatial_ok]
                lat_idx = lat_idx[spatial_ok]
                ages = ages[spatial_ok]
                if ages.size == 0:
                    continue

                total_valid += int(ages.size)
                observed_min_age = min(observed_min_age, float(ages.min()))
                observed_max_age = max(observed_max_age, float(ages.max()))
                older_than_150_count += int(
                    np.count_nonzero(ages >= last_group_upper_label)
                )

                # [0,10), [10,20), ..., [130,140), [140,infinity).
                # Values >= 140 are merged into the 15th class.
                age_idx = np.floor(ages / AGE_STEP).astype(np.int64)
                age_idx = np.minimum(age_idx, N_AGE_GROUPS - 1)

                cell_idx = lat_idx * n_lon + lon_idx
                combined_idx = cell_idx * N_AGE_GROUPS + age_idx

                age_counts += np.bincount(
                    combined_idx, minlength=age_counts.size
                )
                pixel_counts += np.bincount(cell_idx, minlength=n_cells)

                if block_number % 20 == 0 or block_number == total_blocks:
                    print(f"Processed {block_number}/{total_blocks} blocks", flush=True)

    # ========================================================
    # 4. Calculate age probabilities and check the sums
    # ========================================================
    age_counts = age_counts.reshape(n_cells, N_AGE_GROUPS)
    forest_cells = np.flatnonzero(pixel_counts > 0)
    if forest_cells.size == 0:
        raise RuntimeError("No valid forest-age pixels found.")

    counts = age_counts[forest_cells]
    denominators = pixel_counts[forest_cells]
    if not np.array_equal(counts.sum(axis=1), denominators):
        raise RuntimeError("QC failed: age counts do not match pixel totals.")

    probabilities = counts.astype(np.float64) / denominators[:, None]
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("QC failed: probabilities do not sum to one.")

    lat_idx = forest_cells // n_lon
    lon_idx = forest_cells % n_lon
    lon_centers = LON_MIN + (lon_idx + 0.5) * GRID_RES
    lat_centers = LAT_MIN + (lat_idx + 0.5) * GRID_RES

    # ========================================================
    # 5. Export the same table structure as the example
    # ========================================================
    output = pd.DataFrame(probabilities, columns=age_columns)
    output.insert(0, "lat", lat_centers)
    output.insert(0, "lon", lon_centers)
    output["forest_pixel_count"] = denominators
    output["probability_sum"] = probability_sums
    output = output.sort_values(
        ["lat", "lon"], ascending=[False, True]
    ).reset_index(drop=True)
    output.insert(0, "ID", np.arange(1, len(output) + 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False, float_format="%.12f")

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print("Output:", args.output)
    print("Forest grid cells:", len(output))
    print("Valid age pixels:", f"{total_valid:,}")
    print("Excluded negative ages:", f"{negative_count:,}")
    print("Excluded invalid coordinates:", f"{invalid_coordinate_count:,}")
    print("Excluded outside-grid pixels:", f"{outside_grid_count:,}")
    print("Observed age range:", f"{observed_min_age:.2f}–{observed_max_age:.2f}")
    print("Pixels >= 150 merged into final class:", f"{older_than_150_count:,}")
    print("Last class:", age_columns[-1], "means age >=", last_group_start)
    print("Probability sum range:", probability_sums.min(), probability_sums.max())
    print("\nFirst five rows:")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
