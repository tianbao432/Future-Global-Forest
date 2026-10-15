#!/usr/bin/env python3
"""Build one annual forest-type table from two input files."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

TYPE_FILE = Path('/data/global_carbon_sink/ESA_land_cover/forest_type_fractions_0p5deg_2020.txt')
ST_FILE = Path('/data/global_carbon_sink/LC/ForestST_agegroups_1700_2100_15.txt')
OUTPUT_FILE = Path('/data/global_carbon_sink/ESA_land_cover/PFT/Tianbao_ST_PFT_1700_2100.txt')

TYPE_COLS = ['ND', 'BD', 'MX', 'UNK']
ST_COLS = [f'ForestST_{i}_{i + 10}' for i in range(0, 150, 10)]
START_YEAR, END_YEAR = 1700, 2100
NYEARS = END_YEAR - START_YEAR + 1
CHUNK_ROWS = 100000
TOL = 1e-8
SCALE = 10 ** 10


def column_mapping(path, required, optional=()):
    if not path.is_file():
        raise FileNotFoundError(f'Input not found: {path}')
    with path.open(encoding='utf-8-sig') as handle:
        names = next((line.split() for line in handle if line.strip()), [])
    lookup = {name.lower(): name for name in names}
    if len(lookup) != len(names):
        raise ValueError(f'Duplicate column names: {path}')
    missing = [name for name in required if name.lower() not in lookup]
    if missing:
        raise ValueError(f'Missing columns in {path}: {missing}')
    if ST_COLS[0] in required:
        actual = {name for name in lookup if name.startswith('forestst_')}
        if actual != {name.lower() for name in ST_COLS}:
            raise ValueError('Expected exactly 15 ForestST columns: 0_10 through 140_150.')
    selected = list(required) + [name for name in optional if name.lower() in lookup]
    return {lookup[name.lower()]: name for name in selected}


def grid_ids(frame):
    lon = frame['lon'].to_numpy()
    lat = frame['lat'].to_numpy()
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError('Coordinates contain missing or infinite values.')
    if np.any((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)):
        raise ValueError('Coordinates are outside geographic bounds.')
    cols = np.rint((lon + 179.75) / 0.5).astype(np.int64)
    rows = np.rint((89.75 - lat) / 0.5).astype(np.int64)
    valid = (
        (cols >= 0) & (cols < 720) & (rows >= 0) & (rows < 360)
        & (np.abs(lon - (-179.75 + cols * 0.5)) <= TOL)
        & (np.abs(lat - (89.75 - rows * 0.5)) <= TOL)
    )
    if not valid.all():
        raise ValueError('Coordinates must be 0.5-degree grid centers ending in .25 or .75.')
    return rows * 720 + cols


def load_types():
    mapping = column_mapping(TYPE_FILE, ['lon', 'lat', 'ND', 'BD', 'MX'], ['UNK', 'year'])
    frame = pd.read_csv(
        TYPE_FILE, sep=r'\s+', usecols=list(mapping), dtype=np.float64,
        encoding='utf-8-sig').rename(columns=mapping)
    if frame.empty:
        raise ValueError('The type table is empty.')
    if 'year' in frame and not (frame['year'] == 2020).all():
        raise ValueError('The static type table must contain only year 2020.')
    ids = grid_ids(frame)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate coordinates in the type table.')

    if 'UNK' in frame:
        weights = frame[TYPE_COLS].to_numpy(copy=True)
        if not np.isfinite(weights).all():
            raise ValueError('Explicit ND/BD/MX/UNK weights must all be finite.')
    else:
        values = frame[['ND', 'BD', 'MX']].to_numpy()
        finite = np.isfinite(values).all(axis=1)
        all_nan = np.isnan(values).all(axis=1)
        all_zero = finite & (values == 0).all(axis=1)
        if not (finite | all_nan).all():
            raise ValueError('Partial missing values or infinities in ND/BD/MX.')
        unknown = all_nan | all_zero
        weights = np.zeros((len(frame), 4), dtype=np.float64)
        weights[~unknown, :3] = values[~unknown]
        weights[unknown, 3] = 1.0
        print(f'Converted to UNK: all-NaN={all_nan.sum():,}, all-zero={all_zero.sum():,}')

    sums = weights.sum(axis=1)
    invalid = ((weights < 0).any(axis=1) | (weights > 1).any(axis=1)
               | (np.abs(sums - 1) > TOL))
    if invalid.any():
        raise ValueError('Type weights must be in [0, 1] and sum to 1.\n'
                         + frame.loc[invalid].head().to_string(index=False))
    weights /= sums[:, None]
    return frame[['lon', 'lat']].copy(), ids, weights


def split_forest(forest_units, weights):
    """Preserve the four-type total exactly at 10 decimal places."""
    scaled = forest_units[:, None] * weights
    parts = np.floor(scaled).astype(np.int64)
    remainder = forest_units - parts.sum(axis=1)
    if np.any((remainder < 0) | (remainder > 4)):
        raise RuntimeError('Unexpected rounding remainder.')
    order = np.argsort(-(scaled - parts), axis=1, kind='stable')
    for k in range(4):
        rows = np.flatnonzero(remainder > k)
        parts[rows, order[rows, k]] += 1
    if np.any(parts < 0) or not np.array_equal(parts.sum(axis=1), forest_units):
        raise RuntimeError('ND+BD+MX+UNK does not equal FOREST.')
    if np.any((weights == 0) & (parts > 0)) or np.any(np.abs(parts - scaled) > 1.00001):
        raise RuntimeError('A type allocation is inconsistent with its input weight.')
    return parts


def main():
    if OUTPUT_FILE.resolve() in {TYPE_FILE.resolve(), ST_FILE.resolve()}:
        raise ValueError('The output must not overwrite an input file.')
    coords, ids, weights = load_types()
    mapping = column_mapping(ST_FILE, ['lon', 'lat', 'year'] + ST_COLS)
    ngrid = len(coords)
    lookup = np.full(720 * 360, -1, dtype=np.int64)
    lookup[ids] = np.arange(ngrid)
    seen = np.zeros((ngrid, (NYEARS + 7) // 8), dtype=np.uint8)
    counts = np.zeros(ngrid, dtype=np.int64)
    positive_rows = np.zeros(4, dtype=np.int64)
    total = 0

    print(f'Type input: {TYPE_FILE}')
    print(f'ForestST input: {ST_FILE}')
    print(f'Output: {OUTPUT_FILE}')
    print(f'Grid count: {ngrid:,}; expected annual rows: {ngrid * NYEARS:,}')
    print('Input grids with positive weights: ' + ', '.join(
        f'{name}={(weights[:, j] > 0).sum():,}' for j, name in enumerate(TYPE_COLS)), flush=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT_FILE.with_name(OUTPUT_FILE.name + f'.partial.{os.getpid()}')
    last_report = time.monotonic()
    try:
        with temp.open('w', encoding='utf-8', newline='') as handle:
            with pd.read_csv(
                ST_FILE, sep=r'\s+', usecols=list(mapping), dtype=np.float64,
                encoding='utf-8-sig', chunksize=CHUNK_ROWS
            ) as reader:
                for chunk in reader:
                    if chunk.empty:
                        continue
                    chunk = chunk.rename(columns=mapping)
                    idx = lookup[grid_ids(chunk)]
                    if (idx < 0).any():
                        raise ValueError('ForestST coordinates are missing from the type table.\n'
                                         + chunk.loc[idx < 0, ['lon', 'lat']].head().to_string(index=False))

                    y = chunk['year'].to_numpy()
                    if not np.isfinite(y).all() or np.any(
                        (y < START_YEAR) | (y > END_YEAR) | (y != np.rint(y))
                    ):
                        raise ValueError('Years must be integers from 1700 through 2100.')
                    years = y.astype(np.int64)
                    offsets = years - START_YEAR
                    keys = idx * NYEARS + offsets
                    byte = offsets // 8
                    bits = np.left_shift(np.uint8(1), (offsets % 8).astype(np.uint8))
                    if len(np.unique(keys)) != len(keys) or np.any((seen[idx, byte] & bits) != 0):
                        raise ValueError('Duplicate coordinate/year records in ForestST.')
                    np.bitwise_or.at(seen, (idx, byte), bits)
                    counts += np.bincount(idx, minlength=ngrid)

                    stands = chunk[ST_COLS].to_numpy()
                    if not np.isfinite(stands).all() or np.any((stands < 0) | (stands > 1)):
                        raise ValueError('ForestST fractions must be finite and in [0, 1].')
                    forest = stands.sum(axis=1)
                    if np.any(forest > 1 + TOL):
                        raise ValueError('The sum of the 15 ForestST fractions exceeds 1.')
                    forest_units = np.rint(np.clip(forest, 0, 1) * SCALE).astype(np.int64)
                    natural_units = SCALE - forest_units
                    parts = split_forest(forest_units, weights[idx])
                    positive_rows += (parts > 0).sum(axis=0)

                    out = pd.DataFrame({
                        'lon': chunk['lon'].to_numpy(), 'lat': chunk['lat'].to_numpy(),
                        'year': years, 'FOREST': forest_units / SCALE,
                        'NATURAL': natural_units / SCALE})
                    for j, name in enumerate(TYPE_COLS):
                        out[name] = parts[:, j] / SCALE
                    out.to_csv(handle, sep='\t', index=False, header=(total == 0), float_format='%.10f')
                    total += len(out)
                    if time.monotonic() - last_report >= 10:
                        print(f'Processed {total:,} / {ngrid * NYEARS:,} rows.', flush=True)
                        last_report = time.monotonic()

        incomplete = counts != NYEARS
        if incomplete.any():
            example = coords.loc[incomplete].copy()
            example['annual_rows'] = counts[incomplete]
            raise ValueError('Each type-table grid must have all 401 years in ForestST.\n'
                             + example.head().to_string(index=False))
        if total != ngrid * NYEARS:
            raise RuntimeError('Unexpected output row count.')
        os.replace(temp, OUTPUT_FILE)
    finally:
        if temp.exists():
            temp.unlink()

    print('Output rows with positive values: ' + ', '.join(
        f'{name}={positive_rows[j]:,}' for j, name in enumerate(TYPE_COLS)))
    print('PASS: ND+BD+MX+UNK=FOREST; FOREST+NATURAL=1 at 10 decimal places.')
    print(f'SUCCESS: {OUTPUT_FILE}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
