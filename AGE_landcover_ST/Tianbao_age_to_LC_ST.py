#!/usr/bin/env python3
"""Generate consistent 1700-2100 land-cover and 15-ForestST inputs.

Original forest-age probability classes are evaluated BEFORE merging classes
whose lower age bound is 140 years or older. This preserves historical
establishment information when the input has 33 ten-year age classes.

The final ForestST_140_150 is an OPEN-ENDED class representing ages >=140
in the 2020 reference distribution. For 2021-2100, stand area fractions
remain fixed at their 2020 values; the code does not simulate tree growth.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

START_YEAR = 1700
REFERENCE_YEAR = 2020
END_YEAR = 2100
AGE_STEP = 10
CHUNK_GRIDS = 100
DEFAULT_MODE =  "both"

DEFAULT_INPUT = Path("/data/global_carbon_sink/age_groups_0p5deg.txt")
DEFAULT_OUTPUT_DIR = Path("/data/global_carbon_sink/LC")
LC_FILENAME = "forest_landcover_1700_2100_15groups.txt"
ST_FILENAME = "ForestST_agegroups_1700_2100_15.txt"
SUMMARY_FILENAME = "forest_landcover_1700_2100_15groups_summary.txt"

OUTPUT_ST_COLUMNS = [
    f"ForestST_{i}_{i + AGE_STEP}" for i in range(0, 150, AGE_STEP)
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-grids", type=int, default=CHUNK_GRIDS)
    parser.add_argument(
        "--mode", choices=("both", "landcover", "forestst"), default=DEFAULT_MODE
    )
    return parser.parse_args()


def determine_age_columns(columns):
    pattern = re.compile(r"^age_(\d+)_(\d+)$")
    groups = []
    for column in columns:
        match = pattern.fullmatch(str(column))
        if match:
            lower, upper = map(int, match.groups())
            if upper - lower != AGE_STEP:
                raise ValueError(f"Unexpected age-group interval: {column}")
            groups.append((lower, column))
    groups.sort()
    if not groups:
        raise ValueError("No age_*_* probability columns found.")

    lower_bounds = [lower for lower, _ in groups]
    if lower_bounds != list(range(0, lower_bounds[-1] + AGE_STEP, AGE_STEP)):
        raise ValueError("Age-group columns must be contiguous from 0 in steps of 10.")
    if lower_bounds[-1] < 140:
        raise ValueError("The input must contain at least age_140_150.")
    if len(set(lower_bounds)) != len(groups):
        raise ValueError("Duplicate age-group lower bounds detected.")

    return [column for _, column in groups], np.array(lower_bounds, dtype=int)


def read_age_table(input_path):
    if not input_path.is_file():
        raise FileNotFoundError(f"Age-distribution file not found: {input_path}")

    df = pd.read_csv(input_path, sep=r"\s+")
    age_columns, lower_bounds = determine_age_columns(df.columns)
    for column in ["lon", "lat"] + age_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
        df[column] = pd.to_numeric(df[column], errors="coerce")

    coords = df[["lon", "lat"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite longitude or latitude detected.")
    if ((coords[:, 0] < -180) | (coords[:, 0] > 180)).any():
        raise ValueError("Longitude outside [-180, 180] detected.")
    if ((coords[:, 1] < -90) | (coords[:, 1] > 90)).any():
        raise ValueError("Latitude outside [-90, 90] detected.")
    if df.duplicated(["lon", "lat"]).any():
        raise ValueError("Duplicate coordinate pairs detected.")

    probs = df[age_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("Non-finite age probabilities detected.")
    if np.any(probs < -1e-10) or np.any(probs > 1.0 + 1e-10):
        raise ValueError("Age probabilities must lie within [0, 1].")
    probs = np.clip(probs, 0.0, 1.0)
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=0):
        bad = np.flatnonzero(~np.isclose(sums, 1.0, atol=1e-6, rtol=0))
        raise ValueError(
            f"Age probabilities fail sum-to-one QC in {len(bad)} cells; "
            f"first index={bad[0]}, sum={sums[bad[0]]:.10f}"
        )
    probs /= sums[:, None]

    print(f"Input age table: {input_path}")
    print(f"Forest grid cells: {len(df):,}")
    print(f"Original age groups: {len(age_columns)}")
    print(f"Original oldest class: {age_columns[-1]} (open ended)")
    if len(age_columns) == 15:
        print(
            "WARNING: Input has only 15 age groups. Exact age structure above "
            "140 years is lost; forest fractions before 1880 will be zero "
            "under the lower-bound backcasting rule."
        )
    elif lower_bounds[-1] < REFERENCE_YEAR - START_YEAR:
        print(
            "WARNING: Input oldest age-group lower bound is younger than "
            "the 320-year historical lookback. Early years may be forced to zero."
        )
    return coords, probs, lower_bounds


def make_output_chunk(coords, probs, years, inclusion, mode):
    n_cells = coords.shape[0]
    n_years = years.size
    selected = probs @ inclusion.T
    selected = np.clip(selected, 0.0, 1.0)

    if not np.all(np.diff(selected[:, : REFERENCE_YEAR - START_YEAR + 1], axis=1) >= -1e-10):
        raise ValueError("Historical FOREST fraction decreases over time.")
    if not np.allclose(selected[:, REFERENCE_YEAR - START_YEAR], 1.0, atol=1e-8):
        raise ValueError("FOREST fraction in the reference year is not 1.")
    if not np.allclose(selected[:, REFERENCE_YEAR - START_YEAR :], 1.0, atol=1e-8):
        raise ValueError("Future FOREST fractions are not fixed to 2020.")

    lon = np.repeat(coords[:, 0], n_years)
    lat = np.repeat(coords[:, 1], n_years)
    year = np.tile(years, n_cells)

    lc = None
    st = None
    if mode in ("both", "landcover"):
        lc = pd.DataFrame({
            "Lon": lon,
            "Lat": lat,
            "Year": year,
            "NATURAL": (1.0 - selected).ravel(),
            "FOREST": selected.ravel(),
        })

    if mode in ("both", "forestst"):
        # A stand holds the reference-year age cohort, not current-year tree age.
        cube = np.empty((n_cells, n_years, 15), dtype=np.float64)
        cube[:, :, :14] = (
            probs[:, None, :14] * inclusion[None, :, :14]
        )
        # Apply historical presence first; then merge ALL ages >= 140.
        cube[:, :, 14] = probs[:, 14:] @ inclusion[:, 14:].T
        cube = np.clip(cube, 0.0, 1.0)
        stand_sum = cube.sum(axis=2)
        if not np.allclose(stand_sum, selected, atol=1e-8, rtol=0):
            difference = np.max(np.abs(stand_sum - selected))
            raise ValueError(f"ForestST and FOREST totals disagree: {difference}")
        if not np.allclose(
            cube[:, REFERENCE_YEAR - START_YEAR :],
            cube[:, REFERENCE_YEAR - START_YEAR, None, :],
            atol=1e-8, rtol=0,
        ):
            raise ValueError("Future stand fractions must remain at 2020 values.")

        st = pd.DataFrame(
            cube.reshape(n_cells * n_years, 15), columns=OUTPUT_ST_COLUMNS
        )
        st.insert(0, "year", year)
        st.insert(0, "lat", lat)
        st.insert(0, "lon", lon)

    return lc, st, selected.sum(axis=0)


def main():
    args = parse_args()
    if args.chunk_grids < 1:
        raise ValueError("--chunk-grids must be positive.")
    if not (START_YEAR <= REFERENCE_YEAR <= END_YEAR):
        raise ValueError("Invalid model years.")

    coords, probs, lower_bounds = read_age_table(args.input)
    years = np.arange(START_YEAR, END_YEAR + 1, dtype=int)
    thresholds = REFERENCE_YEAR - years
    inclusion = (lower_bounds[None, :] >= thresholds[:, None]).astype(float)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    if args.mode in ("both", "landcover"):
        outputs["landcover"] = args.output_dir / LC_FILENAME
    if args.mode in ("both", "forestst"):
        outputs["forestst"] = args.output_dir / ST_FILENAME

    temps = {key: Path(str(path) + f".partial.{os.getpid()}") for key, path in outputs.items()}
    total_rows = 0
    yearly_sum = np.zeros(years.size, dtype=np.float64)
    print(f"Output mode: {args.mode}")
    print(f"Year range: {START_YEAR}-{END_YEAR}, inclusive ({len(years)} years)")
    print(f"Output ForestST groups: {len(OUTPUT_ST_COLUMNS)} (last represents >=140)")
    for kind, path in outputs.items():
        print(f"{kind}: {path}")

    try:
        for start in range(0, len(coords), args.chunk_grids):
            stop = min(start + args.chunk_grids, len(coords))
            lc_chunk, st_chunk, sums = make_output_chunk(
                coords[start:stop], probs[start:stop], years, inclusion, args.mode
            )
            yearly_sum += sums
            for kind, frame in (("landcover", lc_chunk), ("forestst", st_chunk)):
                if frame is None:
                    continue
                frame.to_csv(
                    temps[kind],
                    sep="\t", index=False, float_format="%.10f",
                    header=(start == 0), mode=("w" if start == 0 else "a"),
                )
            total_rows += (stop - start) * len(years)
            if stop == len(coords) or start == 0 or stop % 1000 == 0:
                print(f"Processed {stop:,}/{len(coords):,} grid cells", flush=True)

        expected_rows = len(coords) * len(years)
        if total_rows != expected_rows:
            raise RuntimeError("Output record-count QC failed.")
        for kind, path in outputs.items():
            if not temps[kind].is_file() or temps[kind].stat().st_size == 0:
                raise RuntimeError(f"Empty output for {kind}")
        for kind, path in outputs.items():
            os.replace(temps[kind], path)

        if args.mode in ("both", "landcover"):
            summary = pd.DataFrame({
                "Year": years,
                "Historical_Age": thresholds,
                "Mean_FOREST": yearly_sum / len(coords),
                "Mean_NATURAL": 1.0 - yearly_sum / len(coords),
            })
            summary.to_csv(
                args.output_dir / SUMMARY_FILENAME,
                sep="\t", index=False, float_format="%.10f"
            )

        print(f"PASS: {total_rows:,} data rows per time-series output")
        print("PASS: all age-probability and land-cover consistency checks")
        print("Years: 1700, 1800, 1880, 2020, 2050, 2100")
        first_lc, first_st, _ = make_output_chunk(
            coords[:1], probs[:1], years, inclusion, "both"
        )
        first_lc = first_lc[first_lc["Year"].isin([1700, 1800, 1880, 2020, 2050, 2100])]
        print(first_lc.to_string(index=False))
    finally:
        for temporary in temps.values():
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    main()
