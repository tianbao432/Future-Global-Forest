#!/bin/bash
#
# Global LPJ-GUESS submission script for ESA_Tianbao_age on COSMOS.
#
# Main changes relative to the 15-point test script:
#   - Uses gridlist2.txt by default.
#   - Uses up to 192 MPI ranks (4 x 48-core COSMOS nodes).
#   - Uses longer walltime for global simulations.
#   - Packs MPI ranks in 48-rank node-sized blocks when possible.
#   - Uses a separate global output directory.
#   - Keeps strict validation and safe merge behavior.
#   - Supports dry-run validation with -t.
#
# IMPORTANT:
#   This script does NOT provide model-state checkpoint/restart inside a rank.
#   If a job times out or fails, the current LPJ-GUESS calculation cannot resume
#   from the exact simulation year without LPJ-GUESS serialization/restart files.
#
# Usage:
#   bash submit_global.sh
#   bash submit_global.sh -t
#   bash submit_global.sh -n job_name
#   bash submit_global.sh -s settings_global.sh
#   bash submit_global.sh -i /path/to/main.ins
#   bash submit_global.sh -o /path/to/output_directory
#

set -euo pipefail

# -----------------------------------------------------------------------------
# Default global-run settings
# -----------------------------------------------------------------------------
REQUESTED_NPROCESS=288
CORES_PER_NODE=48
WALLTIME=24:00:00
WALLTIME_APPEND=4:00:00
PARTITION=lu48

INSFILE=./configs/guess_NFI_init.ins
INPUT_MODULE=cfx
GRIDLIST=./configs/gridlist.txt
OUTFILES='*.out'
EXE=/home/tianbao/learn_code/lpj_build_netcdf/guess
COMPUTEACCOUNT=${COMPUTEACCOUNT:-}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

OUT_DIR="$BASE_DIR/out_put"
name=ESA_Tianbao_age_global
submit_vars_file=""
ins=""
dry_run=0
output_override=""

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'USAGE'
Usage: bash submit_global.sh [-n name] [-s settings.sh] [-i main.ins] [-o output_dir] [-t]

Options:
  -n NAME       Slurm job name
  -s FILE       Source a settings file that overrides defaults
  -i FILE       Override instruction file
  -o DIR        Override output directory
  -t            Validate setup and preview splitting only; do not submit
  -h            Show this help
USAGE
}

while getopts ':n:s:i:o:th' opt; do
    case "$opt" in
        n) name=$OPTARG ;;
        s) submit_vars_file=$OPTARG ;;
        i) ins=$OPTARG ;;
        o) output_override=$OPTARG ;;
        t) dry_run=1 ;;
        h) usage; exit 0 ;;
        :) fail "Option -$OPTARG requires a value." ;;
        \?) fail "Unknown option: -$OPTARG" ;;
    esac
done
shift "$((OPTIND - 1))"
(( $# == 0 )) || fail 'Unexpected positional arguments.'

# Settings file may override REQUESTED_NPROCESS, WALLTIME, WALLTIME_APPEND,
# PARTITION, INSFILE, INPUT_MODULE, GRIDLIST, OUTFILES, EXE, COMPUTEACCOUNT,
# OUT_DIR, and name.
if [[ -n "$submit_vars_file" ]]; then
    [[ -r "$submit_vars_file" ]] || fail "Cannot read settings: $submit_vars_file"
    # shellcheck disable=SC1090
    source "$submit_vars_file"
fi

[[ -z "$ins" ]] || INSFILE=$ins
[[ -z "$output_override" ]] || OUT_DIR=$output_override

[[ "$REQUESTED_NPROCESS" =~ ^[1-9][0-9]*$ ]] || \
    fail 'REQUESTED_NPROCESS must be a positive integer.'
[[ "$CORES_PER_NODE" =~ ^[1-9][0-9]*$ ]] || \
    fail 'CORES_PER_NODE must be a positive integer.'
[[ -n "$name" && -n "$OUTFILES" ]] || \
    fail 'Job name and OUTFILES must not be empty.'

[[ -s "$INSFILE" ]] || fail "Missing or empty instruction file: $INSFILE"
[[ -s "$GRIDLIST" ]] || fail "Missing or empty gridlist: $GRIDLIST"
[[ -x "$EXE" ]] || fail "Executable not found or not executable: $EXE"

INSFILE=$(readlink -f "$INSFILE")
GRIDLIST=$(readlink -f "$GRIDLIST")
EXE=$(readlink -f "$EXE")
OUT_DIR=$(readlink -m "$OUT_DIR")
GRIDLIST_FILENAME=gridlist.txt

# -----------------------------------------------------------------------------
# Validate gridlist and count unique coordinate records
# -----------------------------------------------------------------------------
tmp_grid=$(mktemp)
trap 'rm -f -- "$tmp_grid"' EXIT

awk '
BEGIN {
    num="^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$"
}
{
    sub(/^\357\273\277/, "")
    sub(/\r$/, "")
}
/^[[:space:]]*$/ || /^[[:space:]]*[#!]/ { next }
tolower($1)=="lon" && tolower($2)=="lat" { next }
{
    if (NF<2 || $1 !~ num || $2 !~ num ||
        $1+0 < -180 || $1+0 > 180 || $2+0 < -90 || $2+0 > 90) {
        printf "ERROR: Invalid grid coordinate at line %d: %s\n", NR, $0 > "/dev/stderr"
        exit 1
    }

    key=sprintf("%.8f,%.8f", $1+0, $2+0)
    if (seen[key]++) {
        printf "ERROR: Duplicate grid coordinate at line %d: %s\n", NR, $0 > "/dev/stderr"
        exit 1
    }

    print
}' "$GRIDLIST" > "$tmp_grid" || fail 'Gridlist validation failed.'

mapfile -t GRID_ROWS < "$tmp_grid"
Ngridcells=${#GRID_ROWS[@]}
(( Ngridcells > 0 )) || fail 'No valid grid coordinates were found.'

# -----------------------------------------------------------------------------
# Choose MPI rank count
#
# For >=48 ranks, keep the count in full 48-core COSMOS node blocks.
# If the grid is smaller than one node, use one rank per gridcell.
# -----------------------------------------------------------------------------
NPROCESS=$REQUESTED_NPROCESS

if (( NPROCESS > Ngridcells )); then
    NPROCESS=$Ngridcells
fi

if (( NPROCESS >= CORES_PER_NODE )); then
    NPROCESS=$(( (NPROCESS / CORES_PER_NODE) * CORES_PER_NODE ))
fi

(( NPROCESS > 0 )) || fail 'Calculated NPROCESS is zero.'

NNODES=$(( (NPROCESS + CORES_PER_NODE - 1) / CORES_PER_NODE ))

if (( NPROCESS >= CORES_PER_NODE && NPROCESS % CORES_PER_NODE != 0 )); then
    fail "Internal error: NPROCESS=$NPROCESS is not a multiple of $CORES_PER_NODE."
fi

# -----------------------------------------------------------------------------
# Refuse to overwrite previous results
# -----------------------------------------------------------------------------
if [[ -e "$OUT_DIR" || -L "$OUT_DIR" ]]; then
    [[ -d "$OUT_DIR" && ! -L "$OUT_DIR" ]] || \
        fail "Output path is not a regular directory: $OUT_DIR"

    [[ -z "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
        fail "Output directory is not empty: $OUT_DIR. Preserve/rename it before a new run."
fi

printf '\nProject: %s\n' "$BASE_DIR"
printf 'Instruction file: %s\n' "$INSFILE"
printf 'Gridlist: %s\n' "$GRIDLIST"
printf 'Output: %s\n' "$OUT_DIR"
printf 'Gridcells: %s\n' "$Ngridcells"
printf 'Requested MPI processes: %s\n' "$REQUESTED_NPROCESS"
printf 'MPI processes used: %s\n' "$NPROCESS"
printf 'COSMOS nodes requested: %s\n' "$NNODES"
printf 'Walltime: %s\n' "$WALLTIME"
printf 'Merge walltime: %s\n' "$WALLTIME_APPEND"

q=$((Ngridcells / NPROCESS))
r=$((Ngridcells % NPROCESS))

min_per_rank=$q
max_per_rank=$q
(( r == 0 )) || max_per_rank=$((q + 1))

printf 'Gridcells per rank: %d-%d\n' "$min_per_rank" "$max_per_rank"

# Avoid printing hundreds of nearly identical lines for global runs.
if (( NPROCESS <= 48 )); then
    for ((i=1; i<=NPROCESS; i++)); do
        count=$q
        if (( i <= r )); then count=$((count + 1)); fi
        printf '  run%d: %d gridcell(s)\n' "$i" "$count"
    done
else
    printf 'Preview first 10 ranks:\n'
    for ((i=1; i<=10; i++)); do
        count=$q
        if (( i <= r )); then count=$((count + 1)); fi
        printf '  run%d: %d gridcell(s)\n' "$i" "$count"
    done
    printf '  ...\n'
    printf '  run%d: %d gridcell(s)\n' "$NPROCESS" "$q"
fi

if (( dry_run )); then
    echo
    echo 'CHECK ONLY: No output folders created and no jobs submitted.'
    exit 0
fi

command -v sbatch >/dev/null || fail 'sbatch is not available in this shell.'

# -----------------------------------------------------------------------------
# Prepare output and split gridlist
# -----------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
mkdir "$OUT_DIR/.submission_lock" || \
    fail 'Another submission is using this output directory.'

cp "$tmp_grid" "$OUT_DIR/gridlist_used.txt"

cat > "$OUT_DIR/run_metadata.txt" <<META
project=$BASE_DIR
instruction_file=$INSFILE
gridlist=$GRIDLIST
gridcells=$Ngridcells
requested_nprocess=$REQUESTED_NPROCESS
nprocess=$NPROCESS
nodes=$NNODES
cores_per_node=$CORES_PER_NODE
walltime=$WALLTIME
merge_walltime=$WALLTIME_APPEND
partition=$PARTITION
executable=$EXE
META

cd "$OUT_DIR"
offset=0

for ((i=1; i<=NPROCESS; i++)); do
    count=$q
    if (( i <= r )); then count=$((count + 1)); fi

    mkdir "run$i"
    printf '%s\n' "${GRID_ROWS[@]:offset:count}" > "run$i/$GRIDLIST_FILENAME"
    offset=$((offset + count))
done

(( offset == Ngridcells )) || fail 'Internal error while splitting the gridlist.'

if [[ -f "$BASE_DIR/setup_local.sh" ]]; then
    cp "$BASE_DIR/setup_local.sh" ./setup_local.sh
fi

# -----------------------------------------------------------------------------
# Generate compute job
# -----------------------------------------------------------------------------
{
    printf '#!/bin/bash\n'
    printf '#SBATCH --nodes=%d\n' "$NNODES"
    printf '#SBATCH --ntasks=%d\n' "$NPROCESS"
    printf '#SBATCH --cpus-per-task=1\n'

    if (( NPROCESS >= CORES_PER_NODE )); then
        printf '#SBATCH --ntasks-per-node=%d\n' "$CORES_PER_NODE"
    else
        printf '#SBATCH --ntasks-per-node=%d\n' "$NPROCESS"
    fi

    [[ -z "$PARTITION" ]] || printf '#SBATCH --partition=%s\n' "$PARTITION"
    printf '#SBATCH --time=%s\n' "$WALLTIME"
    printf '#SBATCH --exclusive\n'
    printf '#SBATCH --output=slurm-%%j.log\n'

    printf 'NPROCESS=%q\nEXE=%q\nINSFILE=%q\nINPUT_MODULE=%q\nOUT_DIR=%q\n' \
        "$NPROCESS" "$EXE" "$INSFILE" "$INPUT_MODULE" "$OUT_DIR"

    cat <<'JOB'
set -eo pipefail

# COSMOS can retain SoftwareTree/Milan after a normal purge.
module purge
module load gompi/2025a
module load netCDF/4.9.3
set -u

cd "$OUT_DIR"

command -v mpirun >/dev/null || {
    echo 'ERROR: mpirun was not found.' >&2
    exit 1
}

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -f ./setup_local.sh ]]; then
    srun -n "$SLURM_NNODES" -N "$SLURM_NNODES" \
        --ntasks-per-node=1 bash ./setup_local.sh
fi

if [[ -n "${SNIC_TMP:-}" ]]; then
    for ((i=1; i<=NPROCESS; i++)); do
        rm -f "run$i/local"
        ln -s "$SNIC_TMP" "run$i/local"
    done
else
    echo 'INFO: SNIC_TMP is unset; no optional local-storage links were created.'
fi

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-NA}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-NA}"
echo "Start time: $(date -Is)"

mpirun -np "$NPROCESS" --bind-to core "$EXE" \
    -parallel -input "$INPUT_MODULE" "$INSFILE"

echo "End time: $(date -Is)"
JOB
} > guess.cmd

# -----------------------------------------------------------------------------
# Generate merge job
# -----------------------------------------------------------------------------
{
    printf '#!/bin/bash\n'
    printf '#SBATCH --nodes=1\n'
    printf '#SBATCH --ntasks=1\n'
    printf '#SBATCH --cpus-per-task=1\n'
    [[ -z "$PARTITION" ]] || printf '#SBATCH --partition=%s\n' "$PARTITION"
    printf '#SBATCH --time=%s\n' "$WALLTIME_APPEND"
    printf '#SBATCH --output=slurm-%%j.log\n'
    printf 'NPROCESS=%q\nOUTFILES=%q\nOUT_DIR=%q\n' \
        "$NPROCESS" "$OUTFILES" "$OUT_DIR"

    cat <<'APPEND'
set -euo pipefail
cd "$OUT_DIR"
shopt -s nullglob

read -r -a patterns <<< "$OUTFILES"
declare -A names=()

# Every rank must have a nonempty guess.log before merging.
for ((i=1; i<=NPROCESS; i++)); do
    [[ -s "run$i/guess.log" ]] || {
        echo "ERROR: Missing run$i/guess.log" >&2
        exit 1
    }

    found=0
    for pattern in "${patterns[@]}"; do
        for path in "run$i"/$pattern; do
            [[ -s "$path" && -f "$path" ]] || continue
            names["${path##*/}"]=1
            found=1
        done
    done

    (( found )) || {
        echo "ERROR: No nonempty output files in run$i" >&2
        exit 1
    }
done

printf 'file\tranks_with_file\ttotal_ranks\n' > merge_manifest.tsv

while IFS= read -r file; do
    sources=()

    for ((i=1; i<=NPROCESS; i++)); do
        [[ ! -s "run$i/$file" ]] || sources+=("run$i/$file")
    done

    if (( ${#sources[@]} < NPROCESS )); then
        echo "NOTICE: $file exists in ${#sources[@]} of $NPROCESS ranks; see merge_manifest.tsv."
    fi

    awk '
    function norm(s) {
        sub(/\r$/, "", s)
        gsub(/[ \t]+/, " ", s)
        sub(/^ /, "", s)
        sub(/ $/, "", s)
        return s
    }
    BEGIN {
        num="^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$"
    }
    FNR==1 { first_record=1 }
    first_record && NF==0 { next }
    first_record {
        first_record=0
        current_has_header=($1 !~ num)

        if (!initialized) {
            initialized=1
            has_header=current_has_header
            header=norm($0)
            print
            next
        }

        if (current_has_header!=has_header || (has_header && norm($0)!=header)) {
            printf "ERROR: Output headers differ in %s\n", FILENAME > "/dev/stderr"
            exit 1
        }

        if (!has_header) print
        next
    }
    { print }
    ' "${sources[@]}" > "$file.tmp" || {
        rm -f -- "$file.tmp"
        exit 1
    }

    mv -- "$file.tmp" "$file"
    printf '%s\t%d\t%d\n' "$file" "${#sources[@]}" "$NPROCESS" >> merge_manifest.tsv

done < <(printf '%s\n' "${!names[@]}" | LC_ALL=C sort)

: > guess.log
for ((i=1; i<=NPROCESS; i++)); do
    printf '\n===== run%d =====\n' "$i" >> guess.log
    cat "run$i/guess.log" >> guess.log
done

echo "Merged outputs: $OUT_DIR"
echo 'Inspect merge_manifest.tsv, per-rank logs and year/coordinate coverage before analysis.'
APPEND
} > append.cmd

# -----------------------------------------------------------------------------
# Generate progress helper
# -----------------------------------------------------------------------------
{
    printf '#!/bin/bash\ncd %q || exit 1\n' "$OUT_DIR"
    printf 'NPROCESS=%d\n' "$NPROCESS"
    cat <<'PROGRESS'

show_rank() {
    local i=$1
    echo "===== run$i ====="
    if [[ -f "run$i/guess.log" ]]; then
        tail -n 6 "run$i/guess.log"
    else
        echo 'No log yet.'
    fi
}

if (( NPROCESS <= 24 )); then
    for ((i=1; i<=NPROCESS; i++)); do
        show_rank "$i"
    done
else
    for i in 1 2 3 4 5; do
        show_rank "$i"
    done

    echo '===== middle rank ====='
    mid=$(( (NPROCESS + 1) / 2 ))
    show_rank "$mid"

    echo '===== last five ranks ====='
    start=$((NPROCESS - 4))
    for ((i=start; i<=NPROCESS; i++)); do
        show_rank "$i"
    done
fi
PROGRESS
} > progress.sh

chmod +x progress.sh

bash -n guess.cmd
bash -n append.cmd
bash -n progress.sh

# -----------------------------------------------------------------------------
# Submit jobs
# -----------------------------------------------------------------------------
account_args=()
if [[ -n "$COMPUTEACCOUNT" ]]; then
    account_args=(-A "$COMPUTEACCOUNT")
fi

job_response=$(sbatch --parsable "${account_args[@]}" \
    --chdir="$OUT_DIR" -J "$name" guess.cmd)

job_id=${job_response%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || \
    fail "Unexpected sbatch response: $job_response"

printf 'LPJ-GUESS job: %s\n' "$job_id"
printf 'model_job=%s\n' "$job_id" > job_ids.txt

if append_response=$(sbatch --parsable "${account_args[@]}" \
    --chdir="$OUT_DIR" \
    --dependency="afterok:$job_id" \
    -J "ap_${name#bm_}" append.cmd); then

    append_id=${append_response%%;*}
    [[ "$append_id" =~ ^[0-9]+$ ]] || \
        fail "Unexpected sbatch response: $append_response"

    printf 'Merge job: %s\n' "$append_id"
    printf 'merge_job=%s\n' "$append_id" >> job_ids.txt
else
    echo "ERROR: Model job $job_id was submitted, but merge-job submission failed." >&2
    echo 'Do not rerun submit_global.sh; submit append.cmd with the same dependency instead.' >&2
    exit 1
fi

printf 'Output directory: %s\n' "$OUT_DIR"
printf 'Progress command: bash %q/progress.sh\n' "$OUT_DIR"
printf 'Queue command: squeue -j %s,%s\n' "$job_id" "$append_id"
printf 'Accounting command: sacct -j %s,%s --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS\n' \
    "$job_id" "$append_id"
