#!/usr/bin/env bash
# COSMOS LPJ-GUESS global reference run
# Usage: DRY_RUN=1 bash submit_global.sh
# Usage: bash submit_global.sh

set -Eeuo pipefail

# =============================================================================
# Paths
# =============================================================================

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INS="$BASE/configs/custom_info.ins"
GRID="$BASE/input/gridlist.txt"

EXE="/home/tianbao/learn_code/lpj_build_netcdf/guess"

OUT="$BASE/output"


# =============================================================================
# Global MPI settings
# =============================================================================

# COSMOS lu48:
# 48 CPU cores per node
# 4 nodes x 48 ranks = 192 MPI ranks

NPROCESS=480
CORES_PER_NODE=48
NODES=$(( (NPROCESS + CORES_PER_NODE - 1) / CORES_PER_NODE ))

PARTITION="${PARTITION:-lu48}"

WALLTIME="${WALLTIME:-24:00:00}"
APPEND_TIME="${APPEND_TIME:-04:00:00}"


# =============================================================================
# Check input files
# =============================================================================

for f in "$INS" "$GRID" "$EXE"; do

    [[ -f "$f" ]] || {
        echo "ERROR: Missing file: $f" >&2
        exit 1
    }

done


[[ -x "$EXE" ]] || {
    echo "ERROR: Executable is not runnable: $EXE" >&2
    exit 1
}


[[ "$NPROCESS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: Invalid NPROCESS" >&2
    exit 1
}


# =============================================================================
# Count gridcells
# =============================================================================

NGRID=$(awk '
NF >= 2 &&
$1 ~ /^[-+]?[0-9]+([.][0-9]+)?$/ &&
$2 ~ /^[-+]?[0-9]+([.][0-9]+)?$/ {
    n++
}
END {
    print n+0
}
' "$GRID")


(( NGRID >= NPROCESS )) || {
    echo "ERROR: $NGRID gridcells for $NPROCESS processes"
    exit 1
}


Q=$((NGRID / NPROCESS))
R=$((NGRID % NPROCESS))


echo "============================================================"
echo "GLOBAL LPJ-GUESS REFERENCE RUN"
echo "============================================================"
echo "BASE             = $BASE"
echo "INS              = $INS"
echo "GRID             = $GRID"
echo "OUTPUT           = $OUT"
echo "GRIDCELLS        = $NGRID"
echo "MPI_TASKS        = $NPROCESS"
echo "CORES_PER_NODE   = $CORES_PER_NODE"
echo "NODES            = $NODES"
echo "PARTITION        = $PARTITION"
echo "WALLTIME         = $WALLTIME"
echo "APPEND_TIME      = $APPEND_TIME"
echo "CELLS_PER_RANK   = $Q-$((Q + (R > 0 ? 1 : 0)))"
echo "============================================================"


# =============================================================================
# Prevent overwriting existing results
# =============================================================================

if [[ "${DRY_RUN:-0}" != "1" ]]; then

    command -v sbatch >/dev/null || {
        echo "ERROR: sbatch not found." >&2
        exit 1
    }

    if [[ -d "$OUT" &&
          -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then

        echo "ERROR: $OUT is not empty."
        echo "Move or clear previous outputs first."
        exit 1

    fi

fi


# =============================================================================
# Prepare runtime configuration
# =============================================================================

RUNTIME="$BASE/configs/.ref_runtime_$(date +%Y%m%d_%H%M%S)_$$.ins"


python3 - "$INS" "$RUNTIME" "${DRY_RUN:-0}" <<'PY'

from pathlib import Path
import re
import sys


src, dst = map(Path, sys.argv[1:3])

text = src.read_text(
    encoding="utf-8"
)


# -------------------------------------------------------------------------
# file_gridlist
# -------------------------------------------------------------------------

pattern = re.compile(
    r'(?m)^[ \t]*param[ \t]+"file_gridlist"'
    r'[ \t]*\(\s*str\s*"[^"\n]*"\s*\)'
)


if len(pattern.findall(text)) > 1:
    sys.exit(
        "ERROR: Multiple file_gridlist declarations"
    )


if pattern.search(text):

    text = pattern.sub(
        'param "file_gridlist" (str "./gridlist.txt")',
        text
    )

elif re.search(
    r'(?m)^[ \t]*param[ \t]+"file_gridlist"',
    text
):

    sys.exit(
        "ERROR: Unrecognized file_gridlist declaration"
    )

else:

    text += (
        '\n'
        'param "file_gridlist" '
        '(str "./gridlist.txt")\n'
    )


# -------------------------------------------------------------------------
# outputdirectory
# -------------------------------------------------------------------------

out = re.compile(
    r'(?m)^[ \t]*outputdirectory[ \t]+"[^"\n]*"'
)


if out.search(text):

    text = out.sub(
        'outputdirectory "./"',
        text
    )

else:

    text += (
        '\n'
        'outputdirectory "./"\n'
    )


# -------------------------------------------------------------------------
# Write runtime file
# -------------------------------------------------------------------------

if sys.argv[3] != "1":

    dst.write_text(
        text,
        encoding="utf-8"
    )

PY


# =============================================================================
# Dry run
# =============================================================================

if [[ "${DRY_RUN:-0}" == "1" ]]; then

    echo ""
    echo "DRY_RUN: No jobs submitted."

    echo ""
    echo "Expected split:"
    echo "  $R ranks receive $((Q + 1)) gridcells"
    echo "  $((NPROCESS - R)) ranks receive $Q gridcells"

    exit 0

fi


# =============================================================================
# Prepare output directory
# =============================================================================

mkdir -p "$OUT"


# Keep only valid coordinate rows

awk '
NF >= 2 &&
$1 ~ /^[-+]?[0-9]+([.][0-9]+)?$/ &&
$2 ~ /^[-+]?[0-9]+([.][0-9]+)?$/ {
    print
}
' "$GRID" > "$OUT/gridlist_all.txt"


# =============================================================================
# Split gridcells across MPI processes
# =============================================================================

echo ""
echo "Splitting $NGRID gridcells into $NPROCESS blocks..."


start=1


for ((i=1; i<=NPROCESS; i++)); do

    count=$Q

    if (( i <= R )); then
        count=$((Q + 1))
    fi


    mkdir -p "$OUT/run$i"


    sed -n \
        "${start},$((start + count - 1))p" \
        "$OUT/gridlist_all.txt" \
        > "$OUT/run$i/gridlist.txt"


    start=$((start + count))

done


# =============================================================================
# Verify split
# =============================================================================

TOTAL_SPLIT=$(cat "$OUT"/run*/gridlist.txt | wc -l)


if (( TOTAL_SPLIT != NGRID )); then

    echo "ERROR: Grid split failed."
    echo "Original gridcells : $NGRID"
    echo "Split gridcells    : $TOTAL_SPLIT"

    exit 1

fi


echo "Split completed successfully."
echo "Total split cells: $TOTAL_SPLIT"


echo ""
echo "First 10 blocks:"

for ((i=1; i<=10; i++)); do

    n=$(wc -l < "$OUT/run$i/gridlist.txt")

    echo "  run$i : $n gridcells"

done


echo "  ..."
echo "  run$NPROCESS : $(wc -l < "$OUT/run$NPROCESS/gridlist.txt") gridcells"


# =============================================================================
# Generate main SLURM job
# =============================================================================

cat > "$OUT/guess.cmd" <<EOF
#!/usr/bin/env bash

#SBATCH --job-name=Ref_global
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$NPROCESS
#SBATCH --ntasks-per-node=$CORES_PER_NODE
#SBATCH --cpus-per-task=1
#SBATCH --partition=$PARTITION
#SBATCH --time=$WALLTIME
#SBATCH --exclusive
#SBATCH --chdir=$OUT
#SBATCH --output=$OUT/slurm-%j.log


set -Eeuo pipefail


# =============================================================================
# Environment
# =============================================================================

module purge &>/dev/null

module load gompi/2025a
module load netCDF/4.9.3


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1


echo "============================================================"
echo "REF GLOBAL LPJ-GUESS"
echo "============================================================"
echo "Job ID        : \${SLURM_JOB_ID:-NA}"
echo "Nodes         : \${SLURM_JOB_NUM_NODES:-NA}"
echo "MPI tasks     : \${SLURM_NTASKS:-NA}"
echo "Start time    : \$(date -Is)"
echo "============================================================"


# =============================================================================
# Local temporary links
# =============================================================================

for ((i=1; i<=$NPROCESS; i++)); do

    if [[ -n "\${SNIC_TMP:-}" ]]; then

        rm -f "run\$i/local"

        ln -sfn \
            "\$SNIC_TMP" \
            "run\$i/local"

    fi

done


# =============================================================================
# Run LPJ-GUESS
# =============================================================================

mpirun \
    -np "$NPROCESS" \
    --bind-to core \
    "$EXE" \
    -parallel \
    -input cfx \
    "$RUNTIME"


echo ""
echo "============================================================"
echo "LPJ-GUESS completed."
echo "End time: \$(date -Is)"
echo "============================================================"

EOF


# =============================================================================
# Generate output merging job
# =============================================================================

cat > "$OUT/append.cmd" <<EOF
#!/usr/bin/env bash

#SBATCH --job-name=Ref_append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=$PARTITION
#SBATCH --time=$APPEND_TIME
#SBATCH --chdir=$OUT
#SBATCH --output=$OUT/slurm-%j.log


set -Eeuo pipefail

shopt -s nullglob


echo "============================================================"
echo "MERGING GLOBAL REF OUTPUT"
echo "============================================================"
echo "Start time: \$(date -Is)"


# =============================================================================
# Output files
# =============================================================================

files=(run1/*.out)


((\${#files[@]})) || {

    echo "ERROR: No run1 outputs"

    exit 1

}


echo "Output files found in run1: \${#files[@]}"


# =============================================================================
# Merge each output
# =============================================================================

for first in "\${files[@]}"; do

    name="\${first##*/}"


    echo ""
    echo "Merging: \$name"


    cp \
        "\$first" \
        "\$name"


    for ((i=2; i<=$NPROCESS; i++)); do

        f="run\$i/\$name"


        [[ -f "\$f" ]] || {

            echo "ERROR: Missing \$f"

            exit 1

        }


        awk '
        NR==1 && \$1 !~ /^[-+]?[0-9]/ {
            next
        }
        {
            print
        }
        ' "\$f" >> "\$name"

    done

done


# =============================================================================
# Merge logs
# =============================================================================

logs=(run*/guess.log)


if ((\${#logs[@]})); then

    : > guess.log


    for ((i=1; i<=$NPROCESS; i++)); do

        if [[ -f "run\$i/guess.log" ]]; then

            echo "" >> guess.log
            echo "===== run\$i =====" >> guess.log

            cat \
                "run\$i/guess.log" \
                >> guess.log

        fi

    done

fi


# =============================================================================
# Clean runtime configuration
# =============================================================================

rm -f "$RUNTIME"


echo ""
echo "============================================================"
echo "Merge completed."
echo "End time: \$(date -Is)"
echo "============================================================"

EOF


# =============================================================================
# Generate progress helper
# =============================================================================

cat > "$OUT/progress.sh" <<EOF
#!/usr/bin/env bash

cd "$OUT" || exit 1


echo "============================================================"
echo "REF GLOBAL RUN PROGRESS"
echo "============================================================"


for i in 1 2 3 4 5; do

    echo ""
    echo "===== run\$i ====="

    if [[ -f "run\$i/guess.log" ]]; then

        tail -n 6 \
            "run\$i/guess.log"

    else

        echo "No log yet."

    fi

done


MID=$((NPROCESS / 2))


echo ""
echo "===== run\$MID ====="

if [[ -f "run\$MID/guess.log" ]]; then

    tail -n 6 \
        "run\$MID/guess.log"

else

    echo "No log yet."

fi


START=$((NPROCESS - 4))


for ((i=START; i<=NPROCESS; i++)); do

    echo ""
    echo "===== run\$i ====="

    if [[ -f "run\$i/guess.log" ]]; then

        tail -n 6 \
            "run\$i/guess.log"

    else

        echo "No log yet."

    fi

done

EOF


chmod +x \
    "$OUT/progress.sh"


# =============================================================================
# Verify generated scripts
# =============================================================================

bash -n \
    "$OUT/guess.cmd"

bash -n \
    "$OUT/append.cmd"

bash -n \
    "$OUT/progress.sh"


# =============================================================================
# Submit jobs
# =============================================================================

cd "$OUT"


ACCOUNT=()


if [[ -n "${COMPUTEACCOUNT:-}" ]]; then

    ACCOUNT=(
        --account="$COMPUTEACCOUNT"
    )

fi


JOB=$(

    sbatch \
        --parsable \
        "${ACCOUNT[@]}" \
        guess.cmd

)


JOB="${JOB%%;*}"


APPEND=$(

    sbatch \
        --parsable \
        "${ACCOUNT[@]}" \
        --dependency="afterok:$JOB" \
        append.cmd

)


APPEND="${APPEND%%;*}"


# =============================================================================
# Final information
# =============================================================================

echo ""
echo "============================================================"
echo "GLOBAL REF JOBS SUBMITTED"
echo "============================================================"

echo "Compute job : $JOB"
echo "Merge job   : $APPEND"

echo ""
echo "Output:"
echo "$OUT"

echo ""
echo "Monitor queue:"
echo "squeue -j $JOB,$APPEND"

echo ""
echo "Monitor progress:"
echo "bash $OUT/progress.sh"

echo ""
echo "Accounting:"
echo "sacct -j $JOB,$APPEND --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS"

echo "============================================================"