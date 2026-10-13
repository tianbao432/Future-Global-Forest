# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError as exc:
    raise ImportError(
        "Cartopy is required for this script."
    ) from exc


# ============================================================
# 1. Paths
# ============================================================

INPUT_FILE = Path(
    "/data/global_carbon_sink/Besnard_age_groups_0p5deg_1700.txt"
)

OUTPUT_DIR = Path(
    "/data/global_carbon_sink/age_maps"
)

INDIVIDUAL_DIR = (
    OUTPUT_DIR
    /
    "BENSARD_individual_presence_maps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INDIVIDUAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Age groups
#
# The final class represents all forests with Age >= 140 years.
# ============================================================

AGE_COLUMNS = [

    "age_0_10",
    "age_10_20",
    "age_20_30",
    "age_30_40",
    "age_40_50",

    "age_50_60",
    "age_60_70",
    "age_70_80",
    "age_80_90",
    "age_90_100",

    "age_100_110",
    "age_110_120",
    "age_120_130",
    "age_130_140",
    "age_140_150",

]


AGE_LABELS = {

    "age_0_10":
        "0-10",

    "age_10_20":
        "10-20",

    "age_20_30":
        "20-30",

    "age_30_40":
        "30-40",

    "age_40_50":
        "40-50",

    "age_50_60":
        "50-60",

    "age_60_70":
        "60-70",

    "age_70_80":
        "70-80",

    "age_80_90":
        "80-90",

    "age_90_100":
        "90-100",

    "age_100_110":
        "100-110",

    "age_110_120":
        "110-120",

    "age_120_130":
        "120-130",

    "age_130_140":
        "130-140",

    "age_140_150":
        ">=140",

}


# ============================================================
# 3. General settings
# ============================================================

DPI = 300

PRESENCE_THRESHOLD = 0.0

# One single color for all age-group maps.
MAP_POINT_COLOR = '#DE6E66'

OVERVIEW_POINT_SIZE = 3.4

INDIVIDUAL_POINT_SIZE = 6.0

MAP_POINT_ALPHA = 0.88


# ============================================================
# 4. Global font settings
# ============================================================

plt.rcParams.update({

    "font.family":
        "sans-serif",

    "font.size":
        14,

    "axes.titlesize":
        17,

    "axes.labelsize":
        17,

    "xtick.labelsize":
        13,

    "ytick.labelsize":
        13,

})


# ============================================================
# 5. Check input file
# ============================================================

if not INPUT_FILE.is_file():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


# ============================================================
# 6. Read input table
# ============================================================

print()
print("=" * 80)
print("READING GLOBAL AGE DATA")
print("=" * 80)

print()
print("Input:")
print(INPUT_FILE)


df = pd.read_csv(

    INPUT_FILE,

    sep=r"\s+",

)


print()
print("Number of records:")
print(
    len(df)
)

print()
print("Columns:")
print(
    list(
        df.columns
    )
)


# ============================================================
# 7. Required columns
# ============================================================

required_columns = [

    "lon",
    "lat",

] + AGE_COLUMNS


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise RuntimeError(

        "Missing columns: "

        +

        ", ".join(
            missing_columns
        )

    )


# ============================================================
# 8. Convert columns to numeric
# ============================================================

numeric_columns = [

    "lon",
    "lat",

] + AGE_COLUMNS


if "forest_pixel_count" in df.columns:

    numeric_columns.append(
        "forest_pixel_count"
    )


for column in numeric_columns:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce",

    )


# ============================================================
# 9. Coordinate QC
# ============================================================

valid_coord = (

    df["lon"].notna()

    &

    df["lat"].notna()

)


df = df.loc[
    valid_coord
].copy()


if df.empty:

    raise RuntimeError(
        "No valid coordinate records found."
    )


# ============================================================
# 10. Probability QC
# ============================================================

for column in AGE_COLUMNS:

    if df[column].isna().any():

        raise RuntimeError(
            f"Missing values found in {column}."
        )


    bad = (

        (df[column] < 0.0)

        |

        (df[column] > 1.0)

    )


    if bad.any():

        raise RuntimeError(
            f"Probability outside 0-1 found in {column}."
        )


# ============================================================
# 11. Probability-sum QC
# ============================================================

df["probability_sum_check"] = (

    df[
        AGE_COLUMNS
    ]

    .sum(
        axis=1
    )

)


probability_ok = np.isclose(

    df["probability_sum_check"],

    1.0,

    atol=1e-6,

)


if not probability_ok.all():

    print()
    print(
        "WARNING: Some grid cells do not sum to 1."
    )

    print(

        df.loc[
            ~probability_ok,
            [
                "lon",
                "lat",
                "probability_sum_check",
            ]
        ]

        .head(
            20
        )

        .to_string(
            index=False
        )

    )


# ============================================================
# 12. Duplicate coordinate QC
# ============================================================

duplicate_count = df.duplicated(

    subset=[
        "lon",
        "lat",
    ],

    keep=False,

).sum()


if duplicate_count > 0:

    raise RuntimeError(
        "Duplicate 0.5-degree grid cells detected."
    )


# ============================================================
# 13. Map decoration
# ============================================================

def decorate_map(
    ax,
    small=False,
):

    ax.set_global()


    # Ocean background
    ax.add_feature(

        cfeature.OCEAN,

        facecolor="white",

        zorder=0,

    )


    # Light gray land background
    ax.add_feature(

        cfeature.LAND,

        facecolor="#F3F3F3",

        edgecolor="none",

        zorder=0,

    )


    if small:

        coastline_width = 0.38
        border_width = 0.17

    else:

        coastline_width = 0.60
        border_width = 0.25


    ax.coastlines(

        resolution="110m",

        linewidth=coastline_width,

        color="#555555",

        zorder=3,

    )


    ax.add_feature(

        cfeature.BORDERS,

        linewidth=border_width,

        edgecolor="#A0A0A0",

        zorder=3,

    )


# ============================================================
# 14. Presence summary
# ============================================================

presence_records = []


for column in AGE_COLUMNS:

    present = (

        df[column]
        >
        PRESENCE_THRESHOLD

    )


    n_present = int(
        present.sum()
    )


    presence_records.append({

        "Age_Group":
            column,

        "Age_Label":
            AGE_LABELS[column],

        "N_Total_Forest_Grid_Cells":
            int(
                len(df)
            ),

        "N_Grid_Cells_Present":
            n_present,

        "Fraction_Grid_Cells_Present":
            n_present
            /
            len(df),

    })


presence_summary = pd.DataFrame(
    presence_records
)


PRESENCE_SUMMARY_FILE = (

    OUTPUT_DIR
    /
    "age_group_presence_summary.csv"

)


presence_summary.to_csv(

    PRESENCE_SUMMARY_FILE,

    index=False,

)


# ============================================================
# 15. Individual presence maps
# ============================================================

print()
print("=" * 80)
print("CREATING INDIVIDUAL PRESENCE MAPS")
print("=" * 80)


for i, column in enumerate(
    AGE_COLUMNS
):

    label = AGE_LABELS[
        column
    ]


    present = (

        df[column]
        >
        PRESENCE_THRESHOLD

    )


    x = df.loc[
        present
    ]


    print(
        f"{i + 1:02d}/15  {label}: {len(x):,} cells"
    )


    fig = plt.figure(

        figsize=(
            15.5,
            7.5,
        )

    )


    ax = fig.add_subplot(

        1,
        1,
        1,

        projection=ccrs.Robinson(),

    )


    decorate_map(
        ax,
        small=False,
    )


    ax.scatter(

        x["lon"],

        x["lat"],

        s=INDIVIDUAL_POINT_SIZE,

        color=MAP_POINT_COLOR,

        marker="s",

        linewidths=0,

        alpha=MAP_POINT_ALPHA,

        transform=ccrs.PlateCarree(),

        rasterized=True,

        zorder=2,

    )


    if column == "age_140_150":

        title = (
            "Forest age >=140 years"
        )

    else:

        title = (
            f"Forest age {label} years"
        )


    ax.set_title(

        title,

        fontsize=22,

        fontweight="bold",

        pad=14,

    )


    fig.text(

        0.5,

        0.035,

        (
            f"Occurrence in {len(x):,} "
            "forest 0.5-degree grid cells"
        ),

        ha="center",

        fontsize=12,

        color="#444444",

    )


    PNG_FILE = (

        INDIVIDUAL_DIR
        /
        f"{i + 1:02d}_{column}_presence.png"

    )


    PDF_FILE = (

        INDIVIDUAL_DIR
        /
        f"{i + 1:02d}_{column}_presence.pdf"

    )


    fig.savefig(

        PNG_FILE,

        dpi=DPI,

        bbox_inches="tight",

        facecolor="white",

    )


    fig.savefig(

        PDF_FILE,

        bbox_inches="tight",

        facecolor="white",

    )


    plt.close(
        fig
    )


# ============================================================
# 16. Global presence overview
#
# Layout:
#   3 rows x 5 columns
# ============================================================

print()
print("=" * 80)
print("CREATING 3 x 5 PRESENCE OVERVIEW")
print("=" * 80)


fig, axes = plt.subplots(

    3,
    5,

    figsize=(
        30,
        14,
    ),

    subplot_kw={

        "projection":
            ccrs.Robinson()

    },

)


axes = np.asarray(
    axes
).ravel()


for i, column in enumerate(
    AGE_COLUMNS
):

    ax = axes[
        i
    ]


    present = (

        df[column]
        >
        PRESENCE_THRESHOLD

    )


    x = df.loc[
        present
    ]


    decorate_map(
        ax,
        small=True,
    )


    ax.scatter(

        x["lon"],

        x["lat"],

        s=OVERVIEW_POINT_SIZE,

        color=MAP_POINT_COLOR,

        marker="s",

        linewidths=0,

        alpha=MAP_POINT_ALPHA,

        transform=ccrs.PlateCarree(),

        rasterized=True,

        zorder=2,

    )


    panel_letter = chr(

        ord("a")
        +
        i

    )


    if column == "age_140_150":

        title = (
            f"({panel_letter}) >=140 years"
        )

    else:

        title = (
            f"({panel_letter}) "
            f"{AGE_LABELS[column]} years"
        )


    ax.set_title(

        title,

        fontsize=18,

        fontweight="bold",

        pad=8,

    )


# ============================================================
# 17. Overview layout
# ============================================================

fig.subplots_adjust(

    left=0.018,

    right=0.982,

    top=0.90,

    bottom=0.035,

    hspace=0.10,

    wspace=0.025,

)


fig.suptitle(

    (
        "Global occurrence of forest age groups "
        "at 0.5-degree resolution"
    ),

    fontsize=27,

    fontweight="bold",

    y=0.965,

)


OVERVIEW_PNG = (

    OUTPUT_DIR
    /
    "global_age_group_presence_5x3.png"

)


OVERVIEW_PDF = (

    OUTPUT_DIR
    /
    "global_age_group_presence_5x3.pdf"

)


fig.savefig(

    OVERVIEW_PNG,

    dpi=DPI,

    bbox_inches="tight",

    facecolor="white",

)


fig.savefig(

    OVERVIEW_PDF,

    bbox_inches="tight",

    facecolor="white",

)


plt.close(
    fig
)


# ============================================================
# 18. Calculate global forest age probability distribution
#
# Preferred method:
#
#   Probability_i =
#
#       Sum(
#           forest_pixel_count_cell
#           *
#           age_probability_i_cell
#       )
#       -----------------------------------
#       Sum(forest_pixel_count_cell)
#
# This approximates the global distribution using the
# original valid forest-pixel counts.
# ============================================================

print()
print("=" * 80)
print("CALCULATING GLOBAL AGE DISTRIBUTION")
print("=" * 80)


if "forest_pixel_count" in df.columns:

    if df["forest_pixel_count"].isna().any():

        raise RuntimeError(
            "Missing forest_pixel_count values found."
        )


    if (
        df["forest_pixel_count"]
        <=
        0
    ).any():

        raise RuntimeError(
            "forest_pixel_count must be greater than zero."
        )


    forest_weights = df[
        "forest_pixel_count"
    ].to_numpy(
        dtype=float
    )


    print()
    print(
        "Using forest_pixel_count weighting."
    )


else:

    forest_weights = np.ones(

        len(df),

        dtype=float,

    )


    print()
    print(
        "WARNING: forest_pixel_count not found."
    )

    print(
        "Equal weighting of 0.5-degree forest cells is used."
    )


total_weight = float(
    forest_weights.sum()
)


global_probabilities = []


for column in AGE_COLUMNS:

    values = df[
        column
    ].to_numpy(
        dtype=float
    )


    weighted_age_amount = float(

        np.sum(

            values
            *
            forest_weights

        )

    )


    probability = (

        weighted_age_amount
        /
        total_weight

    )


    global_probabilities.append(
        probability
    )


global_probabilities = np.asarray(

    global_probabilities,

    dtype=float,

)


global_probability_sum = float(

    global_probabilities.sum()

)


print()
print(
    "Global probability sum:",
    global_probability_sum
)


# ============================================================
# 19. Global probability output table
# ============================================================

GLOBAL_DISTRIBUTION_TABLE = pd.DataFrame({

    "Age_Group":
        AGE_COLUMNS,

    "Age_Label":
        [
            AGE_LABELS[column]
            for column in AGE_COLUMNS
        ],

    "Probability":
        global_probabilities,

    "Percent":
        global_probabilities
        *
        100.0,

})


GLOBAL_DISTRIBUTION_FILE = (

    OUTPUT_DIR
    /
    "global_forest_age_probability_distribution.csv"

)


GLOBAL_DISTRIBUTION_TABLE.to_csv(

    GLOBAL_DISTRIBUTION_FILE,

    index=False,

)


print()
print(

    GLOBAL_DISTRIBUTION_TABLE.to_string(
        index=False
    )

)


# ============================================================
# 20. Probability histogram
#
# This figure is similar to a conventional age-distribution
# histogram.
#
# Each bar represents one 10-year forest age class.
#
# The last bar represents all Age >= 140 years.
# ============================================================

print()
print("=" * 80)
print("CREATING GLOBAL AGE PROBABILITY HISTOGRAM")
print("=" * 80)


fig, ax = plt.subplots(

    figsize=(
        16,
        9,
    )

)


# ============================================================
# 21. Numeric positions
#
# Mid-points:
#
#   5, 15, 25, ..., 135, 145
# ============================================================

age_midpoints = np.arange(

    5,

    150,

    10,

)


# ============================================================
# 22. Histogram colors
#
# A sequential age gradient is used only for the histogram.
# Younger forests are cooler and older forests are warmer.
# ============================================================

hist_colors = plt.cm.viridis(

    np.linspace(
        0.10,
        0.90,
        len(
            AGE_COLUMNS
        )
    )

)


bars = ax.bar(

    age_midpoints,

    global_probabilities,

    width=9.2,

    color=hist_colors,

    edgecolor="#303030",

    linewidth=0.9,

    align="center",

)


# ============================================================
# 23. Histogram axis formatting
# ============================================================

ax.set_xlim(
    0,
    152
)


ax.set_xticks(
    age_midpoints
)


ax.set_xticklabels(

    [
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        "60-70",
        "70-80",
        "80-90",
        "90-100",
        "100-110",
        "110-120",
        "120-130",
        "130-140",
        ">=140",
    ],

    rotation=45,

    ha="right",

    fontsize=13,

)


ax.set_xlabel(

    "Forest age group (years)",

    fontsize=18,

    fontweight="bold",

    labelpad=12,

)


ax.set_ylabel(

    "Global forest proportion",

    fontsize=18,

    fontweight="bold",

    labelpad=12,

)


# IMPORTANT:
# Correct matplotlib argument is xmax, not x_max.

ax.yaxis.set_major_formatter(

    PercentFormatter(
        xmax=1.0,
        decimals=0,
    )

)


ax.tick_params(

    axis="y",

    labelsize=14,

)


ax.grid(

    axis="y",

    linestyle="--",

    linewidth=0.7,

    alpha=0.25,

)


ax.set_axisbelow(
    True
)


# ============================================================
# 24. Histogram title
# ============================================================

ax.set_title(

    "Global forest age probability distribution",

    fontsize=24,

    fontweight="bold",

    pad=16,

)


# ============================================================
# 25. Add percentages above bars
# ============================================================

maximum_probability = float(

    np.max(
        global_probabilities
    )

)


label_offset = max(

    maximum_probability
    *
    0.015,

    0.001,

)


for bar, probability in zip(

    bars,

    global_probabilities,

):

    ax.text(

        bar.get_x()
        +
        bar.get_width()
        /
        2.0,

        probability
        +
        label_offset,

        f"{probability * 100.0:.1f}%",

        ha="center",

        va="bottom",

        fontsize=10.5,

        fontweight="bold",

    )


ax.set_ylim(

    0.0,

    maximum_probability
    *
    1.16,

)


# ============================================================
# 26. Histogram note
# ============================================================

if "forest_pixel_count" in df.columns:

    histogram_note = (
        "Global proportions are weighted by valid forest-pixel "
        "counts within each 0.5-degree grid cell."
    )

else:

    histogram_note = (
        "Global proportions are based on equal weighting of "
        "0.5-degree forest grid cells."
    )


fig.text(

    0.5,

    0.015,

    histogram_note,

    ha="center",

    fontsize=11.5,

    color="#444444",

)


fig.tight_layout(

    rect=[
        0.03,
        0.05,
        0.99,
        0.95,
    ]

)


# ============================================================
# 27. Save histogram
# ============================================================

HISTOGRAM_PNG = (

    OUTPUT_DIR
    /
    "global_forest_age_probability_histogram.png"

)


HISTOGRAM_PDF = (

    OUTPUT_DIR
    /
    "global_forest_age_probability_histogram.pdf"

)


fig.savefig(

    HISTOGRAM_PNG,

    dpi=DPI,

    bbox_inches="tight",

    facecolor="white",

)


fig.savefig(

    HISTOGRAM_PDF,

    bbox_inches="tight",

    facecolor="white",

)


plt.close(
    fig
)


# ============================================================
# 28. Final report
# ============================================================

print()
print("=" * 80)
print("VISUALIZATION FINISHED")
print("=" * 80)

print()
print("Input:")
print(
    INPUT_FILE
)

print()
print("Number of forest grid cells:")
print(
    len(df)
)

print()
print("Individual presence maps:")
print(
    INDIVIDUAL_DIR
)

print()
print("3 x 5 presence overview:")
print(
    OVERVIEW_PNG
)

print()
print("3 x 5 presence overview PDF:")
print(
    OVERVIEW_PDF
)

print()
print("Global probability histogram:")
print(
    HISTOGRAM_PNG
)

print()
print("Global probability histogram PDF:")
print(
    HISTOGRAM_PDF
)

print()
print("Global probability table:")
print(
    GLOBAL_DISTRIBUTION_FILE
)

print()
print("Presence summary:")
print(
    PRESENCE_SUMMARY_FILE
)

print()
print("=" * 80)