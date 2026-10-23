#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global mean net carbon uptake (-NEE) maps
for four LPJ-GUESS simulations.

Period:
    2020-2100

Definition:
    Net carbon uptake = -NEE

Therefore:
    positive = carbon sink
    negative = carbon source
"""

# ============================================================
# 0. HEADLESS BACKEND
# ============================================================

import os

os.environ["MPLBACKEND"] = "Agg"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUTS = {

    "Reference": (
        "/lunarc/nobackup/projects/snic2020-6-23/"
        "tianbao/Ref/output/cflux.out"
    ),

    "Tianbao age": (
        "/lunarc/nobackup/projects/snic2020-6-23/"
        "tianbao/Tianbao_AGE_IN/out_put/cflux.out"
    ),

    "Besnard age": (
        "/lunarc/nobackup/projects/snic2020-6-23/"
        "tianbao/ESA_Besnard_age/out_put/cflux.out"
    ),

    "ESA-Tianbao age": (
        "/lunarc/nobackup/projects/snic2020-6-23/"
        "tianbao/ESA_Tianbao_age/check_result/"
        "ESA_Tianbao/cflux.out"
    ),
}


OUTPUT_DIR = (
    "/lunarc/nobackup/projects/snic2020-6-23/"
    "tianbao/result_plot/NEE_map/"
)


START_YEAR = 2020
END_YEAR = 2100

COLOR_PERCENTILE = 98.0

DPI = 600


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. GLOBAL STYLE
# ============================================================

mpl.rcParams.update({

    "font.family": "sans-serif",

    "font.sans-serif": [
        "Arial",
        "DejaVu Sans"
    ],

    "font.size": 10,

    "axes.titlesize": 14,

    "axes.titleweight": "bold",

    "figure.facecolor": "white",

    "savefig.facecolor": "white",

    "axes.facecolor": "white",

})


# ============================================================
# 3. FIND COLUMN
# ============================================================

def find_column(columns, candidates):

    lower_map = {
        str(col).lower(): col
        for col in columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lower_map:
            return lower_map[key]

    return None


# ============================================================
# 4. READ CFLUX
# ============================================================

def read_cflux(file_path, simulation_name):

    print("\n")
    print("=" * 90)
    print("Simulation :", simulation_name)
    print("Input file :", file_path)
    print("=" * 90)

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            "Input file does not exist:\n{}".format(
                file_path
            )
        )

    # --------------------------------------------------------
    # Read LPJ-GUESS output
    # --------------------------------------------------------

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="python"
    )

    print(
        "Rows read  : {:,}".format(
            len(df)
        )
    )

    print(
        "Columns    :",
        list(df.columns)
    )


    # --------------------------------------------------------
    # Detect columns
    # --------------------------------------------------------

    lon_col = find_column(
        df.columns,
        [
            "Lon",
            "lon",
            "Longitude",
            "longitude",
            "LON",
            "LONGITUDE"
        ]
    )

    lat_col = find_column(
        df.columns,
        [
            "Lat",
            "lat",
            "Latitude",
            "latitude",
            "LAT",
            "LATITUDE"
        ]
    )

    year_col = find_column(
        df.columns,
        [
            "Year",
            "year",
            "YEAR"
        ]
    )

    nee_col = find_column(
        df.columns,
        [
            "NEE",
            "nee",
            "Nee",
            "NEE_kgC",
            "nee_kgc"
        ]
    )


    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    missing = []

    if lon_col is None:
        missing.append("longitude")

    if lat_col is None:
        missing.append("latitude")

    if year_col is None:
        missing.append("year")

    if nee_col is None:
        missing.append("NEE")


    if missing:

        raise ValueError(
            "\nCould not identify required columns: {}\n"
            "Available columns:\n{}\n".format(
                missing,
                list(df.columns)
            )
        )


    print("\nDetected columns:")

    print(
        "  Longitude :",
        lon_col
    )

    print(
        "  Latitude  :",
        lat_col
    )

    print(
        "  Year      :",
        year_col
    )

    print(
        "  NEE       :",
        nee_col
    )


    # --------------------------------------------------------
    # Standard dataframe
    # --------------------------------------------------------

    temp = pd.DataFrame({

        "lon": pd.to_numeric(
            df[lon_col],
            errors="coerce"
        ),

        "lat": pd.to_numeric(
            df[lat_col],
            errors="coerce"
        ),

        "year": pd.to_numeric(
            df[year_col],
            errors="coerce"
        ),

        "NEE": pd.to_numeric(
            df[nee_col],
            errors="coerce"
        ),

    })


    temp = temp.dropna(
        subset=[
            "lon",
            "lat",
            "year",
            "NEE"
        ]
    )


    # --------------------------------------------------------
    # Select 2020-2100
    # --------------------------------------------------------

    temp = temp[
        (temp["year"] >= START_YEAR)
        &
        (temp["year"] <= END_YEAR)
    ].copy()


    if len(temp) == 0:

        raise ValueError(
            "No valid data found during {}-{} for {}".format(
                START_YEAR,
                END_YEAR,
                simulation_name
            )
        )


    # --------------------------------------------------------
    # Convert sign
    # --------------------------------------------------------
    #
    # LPJ-GUESS:
    # positive NEE = release
    # negative NEE = uptake
    #
    # This study:
    # positive = carbon uptake
    # negative = carbon release
    # --------------------------------------------------------

    temp["Net_uptake"] = -temp["NEE"]


    # --------------------------------------------------------
    # Mean over 2020-2100
    # --------------------------------------------------------

    result = (
        temp
        .groupby(
            ["lon", "lat"],
            as_index=False
        )
        .agg(
            Net_uptake=(
                "Net_uptake",
                "mean"
            ),
            N_years=(
                "year",
                "nunique"
            )
        )
    )


    # --------------------------------------------------------
    # QC
    # --------------------------------------------------------

    print(
        "\nPeriod             : {}-{}".format(
            START_YEAR,
            END_YEAR
        )
    )

    print(
        "Rows in period     : {:,}".format(
            len(temp)
        )
    )

    print(
        "Unique grid cells  : {:,}".format(
            len(result)
        )
    )

    print(
        "Mean years/grid    : {:.2f}".format(
            result["N_years"].mean()
        )
    )

    print(
        "Mean uptake        : {:.6f}".format(
            result["Net_uptake"].mean()
        )
    )

    print(
        "Median uptake      : {:.6f}".format(
            result["Net_uptake"].median()
        )
    )

    print(
        "Minimum            : {:.6f}".format(
            result["Net_uptake"].min()
        )
    )

    print(
        "Maximum            : {:.6f}".format(
            result["Net_uptake"].max()
        )
    )


    return result


# ============================================================
# 5. READ ALL SIMULATIONS
# ============================================================

datasets = {}


for simulation_name, file_path in INPUTS.items():

    datasets[simulation_name] = read_cflux(
        file_path=file_path,
        simulation_name=simulation_name
    )


# ============================================================
# 6. EXPORT GRID MEAN TABLES
# ============================================================

print("\n")
print("=" * 90)
print("EXPORTING GRID-CELL MEAN TABLES")
print("=" * 90)


for simulation_name, df_mean in datasets.items():

    safe_name = (
        simulation_name
        .replace(" ", "_")
        .replace("-", "_")
    )

    csv_file = os.path.join(
        OUTPUT_DIR,
        "{}_mean_net_uptake_{}_{}.csv".format(
            safe_name,
            START_YEAR,
            END_YEAR
        )
    )

    df_mean.to_csv(
        csv_file,
        index=False
    )

    print(csv_file)


# ============================================================
# 7. COMMON COLOR SCALE
# ============================================================

all_values = np.concatenate([

    df_mean[
        "Net_uptake"
    ]
    .dropna()
    .to_numpy()

    for df_mean
    in datasets.values()

])


all_abs_values = np.abs(
    all_values
)


vmax = np.nanpercentile(
    all_abs_values,
    COLOR_PERCENTILE
)


if (
    not np.isfinite(vmax)
    or
    vmax <= 0
):

    vmax = np.nanmax(
        all_abs_values
    )


if (
    not np.isfinite(vmax)
    or
    vmax <= 0
):

    vmax = 1.0


vmin = -vmax


print("\n")
print("=" * 90)
print("COMMON COLOR SCALE")
print("=" * 90)

print(
    "Percentile used : {:.1f}%".format(
        COLOR_PERCENTILE
    )
)

print(
    "vmin            : {:.6f}".format(
        vmin
    )
)

print(
    "vmax            : {:.6f}".format(
        vmax
    )
)


# ============================================================
# 8. COLOR MAP
# ============================================================

# Carbon source:
# plum / warm brown
#
# Neutral:
# warm off-white
#
# Carbon sink:
# turquoise / deep teal


colors = [

    "#5B2333",

    "#82403B",

    "#A95F3B",

    "#D08B48",

    "#E4BC78",

    "#F2E4BC",

    "#F7F6F0",

    "#DDEDEA",

    "#B8DED8",

    "#83CCC2",

    "#48B3A9",

    "#198B86",

    "#08676B",

    "#084C61",

]


cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "carbon_balance",
    colors,
    N=256
)


norm = mpl.colors.TwoSlopeNorm(

    vmin=vmin,

    vcenter=0.0,

    vmax=vmax

)


# ============================================================
# 9. PANEL FUNCTION
# ============================================================

def plot_panel(
    ax,
    data,
    title,
    panel_label
):

    ax.set_global()


    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax.set_facecolor(
        "#F8FAFA"
    )


    ax.add_feature(
        cfeature.OCEAN,
        facecolor="#F8FAFA",
        edgecolor="none",
        zorder=0
    )


    ax.add_feature(
        cfeature.LAND,
        facecolor="#F0F3F2",
        edgecolor="none",
        zorder=0
    )


    # --------------------------------------------------------
    # Forest grid cells
    # --------------------------------------------------------

    sc = ax.scatter(

        data["lon"],

        data["lat"],

        c=data[
            "Net_uptake"
        ],

        cmap=cmap,

        norm=norm,

        s=3.6,

        marker="s",

        linewidths=0,

        rasterized=True,

        transform=ccrs.PlateCarree(),

        zorder=2
    )


    # --------------------------------------------------------
    # Coastline
    # --------------------------------------------------------

    ax.coastlines(
        resolution="110m",
        linewidth=0.42,
        color="#6D7976",
        zorder=3
    )


    # --------------------------------------------------------
    # Borders
    # --------------------------------------------------------

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.15,
        edgecolor="#A7B0AD",
        zorder=3
    )


    # --------------------------------------------------------
    # Very subtle graticules
    # --------------------------------------------------------

    ax.gridlines(

        crs=ccrs.PlateCarree(),

        draw_labels=False,

        linewidth=0.22,

        color="#B8C2BF",

        alpha=0.16,

        linestyle="-",

        zorder=1
    )


    # --------------------------------------------------------
    # Map boundary
    # --------------------------------------------------------

    try:

        ax.spines[
            "geo"
        ].set_linewidth(
            0.65
        )

        ax.spines[
            "geo"
        ].set_edgecolor(
            "#4F5956"
        )

    except Exception:

        pass


    # --------------------------------------------------------
    # Panel label
    #
    # Put inside map to avoid layout expansion
    # --------------------------------------------------------

    ax.text(

        0.025,

        0.935,

        panel_label,

        transform=ax.transAxes,

        ha="left",

        va="top",

        fontsize=13.5,

        fontweight="bold",

        color="#18201E",

        zorder=10

    )


    # --------------------------------------------------------
    # Panel title
    # --------------------------------------------------------

    ax.set_title(

        title,

        fontsize=14.0,

        fontweight="bold",

        pad=6,

        color="#17201E"

    )


    return sc


# ============================================================
# 10. FIGURE
# ============================================================

projection = ccrs.Robinson(
    central_longitude=0
)


# Slightly wider than tall
fig = plt.figure(
    figsize=(16.2, 9.6),
    facecolor="white"
)


# ============================================================
# 11. MANUAL GRID LAYOUT
# ============================================================

# This layout explicitly reserves:
#
# top    : main title
# middle : four maps
# bottom : colorbar
#
# Manual positions prevent title overlap.

gs = fig.add_gridspec(

    nrows=2,

    ncols=2,

    left=0.035,

    right=0.965,

    bottom=0.145,

    top=0.885,

    wspace=0.070,

    hspace=0.105

)


axes = [

    fig.add_subplot(
        gs[0, 0],
        projection=projection
    ),

    fig.add_subplot(
        gs[0, 1],
        projection=projection
    ),

    fig.add_subplot(
        gs[1, 0],
        projection=projection
    ),

    fig.add_subplot(
        gs[1, 1],
        projection=projection
    ),

]


# ============================================================
# 12. PANEL ORDER
# ============================================================

plot_order = [

    (
        "Reference",
        "(a)"
    ),

    (
        "Tianbao age",
        "(b)"
    ),

    (
        "Besnard age",
        "(c)"
    ),

    (
        "ESA-Tianbao age",
        "(d)"
    ),

]


last_scatter = None


for ax, (
    simulation_name,
    panel_label
) in zip(
    axes,
    plot_order
):

    last_scatter = plot_panel(

        ax=ax,

        data=datasets[
            simulation_name
        ],

        title=simulation_name,

        panel_label=panel_label

    )


# ============================================================
# 13. MAIN TITLE
# ============================================================

# Only one title.
# No subtitle or small explanatory text.

fig.suptitle(

    "Mean net carbon uptake (-NEE), 2020-2100",

    fontsize=19,

    fontweight="bold",

    x=0.5,

    y=0.958,

    ha="center",

    va="center",

    color="#16201D"

)


# ============================================================
# 14. COMMON COLORBAR
# ============================================================

# Lower position leaves clean separation from second map row.

cbar_ax = fig.add_axes(
    [
        0.27,    # left
        0.075,   # bottom
        0.46,    # width
        0.020    # height
    ]
)


cbar = fig.colorbar(

    last_scatter,

    cax=cbar_ax,

    orientation="horizontal",

    extend="both"

)


# ============================================================
# 15. COLORBAR TICKS
# ============================================================

ticks = np.linspace(
    vmin,
    vmax,
    5
)


cbar.set_ticks(
    ticks
)


cbar.set_ticklabels(
    [
        "{:.3f}".format(x)
        for x in ticks
    ]
)


cbar.ax.tick_params(

    labelsize=9.5,

    width=0.5,

    length=3,

    pad=3

)


cbar.set_label(

    "Net carbon uptake (kg C m$^{-2}$ yr$^{-1}$)",

    fontsize=11.5,

    labelpad=7

)


cbar.outline.set_linewidth(
    0.55
)


cbar.outline.set_edgecolor(
    "#515A57"
)


# ============================================================
# 16. SAVE FIGURE
# ============================================================

png_file = os.path.join(
    OUTPUT_DIR,
    "NEE_global_mean_2020_2100_four_simulations_final.png"
)


pdf_file = os.path.join(
    OUTPUT_DIR,
    "NEE_global_mean_2020_2100_four_simulations_final.pdf"
)


svg_file = os.path.join(
    OUTPUT_DIR,
    "NEE_global_mean_2020_2100_four_simulations_final.svg"
)


fig.savefig(

    png_file,

    dpi=DPI,

    bbox_inches="tight",

    pad_inches=0.12

)


fig.savefig(

    pdf_file,

    bbox_inches="tight",

    pad_inches=0.12

)


fig.savefig(

    svg_file,

    bbox_inches="tight",

    pad_inches=0.12

)


plt.close(fig)


# ============================================================
# 17. SUMMARY STATISTICS
# ============================================================

summary_rows = []


for simulation_name, df_mean in datasets.items():

    values = (
        df_mean[
            "Net_uptake"
        ]
        .dropna()
    )


    summary_rows.append({

        "Simulation":
            simulation_name,

        "N_grid":
            len(df_mean),

        "Mean":
            values.mean(),

        "Median":
            values.median(),

        "SD":
            values.std(),

        "Min":
            values.min(),

        "Max":
            values.max(),

        "P05":
            values.quantile(
                0.05
            ),

        "P95":
            values.quantile(
                0.95
            ),

        "Sink_fraction_pct":
            100.0
            *
            np.mean(
                values > 0
            ),

        "Source_fraction_pct":
            100.0
            *
            np.mean(
                values < 0
            ),

    })


summary = pd.DataFrame(
    summary_rows
)


summary_file = os.path.join(

    OUTPUT_DIR,

    "NEE_global_mean_2020_2100_summary.csv"

)


summary.to_csv(

    summary_file,

    index=False

)


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)


print(

    summary.to_string(

        index=False,

        float_format=lambda x:
        "{:.5f}".format(x)

    )

)


print("\n")
print("=" * 90)
print("OUTPUT FILES")
print("=" * 90)


print(
    "\nPNG:\n{}".format(
        png_file
    )
)


print(
    "\nPDF:\n{}".format(
        pdf_file
    )
)


print(
    "\nSVG:\n{}".format(
        svg_file
    )
)


print(
    "\nSummary CSV:\n{}".format(
        summary_file
    )
)


print("\nDONE")