#!/bin/bash
# Submit LPJ-GUESS from the ESA_Tianbao_age project directory.
# Usage: bash submit.sh [-n job_name] [-s settings.sh] [-i main.ins] [-t]
# -t validates the setup and previews grid splitting without submitting.

set -euo pipefail

NPROCESS=15
WALLTIME=2:00:00
WALLTIME_APPEND=1:00:00
INSFILE=./configs/guess_NFI_init.ins
INPUT_MODULE=cfx
GRIDLIST=./configs/gridlist.txt
OUTFILES='*.out'
EXE=/home/tianbao/learn_code/lpj_build_netcdf/guess
COMPUTEACCOUNT=${COMPUTEACCOUNT:-}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"
OUT_DIR="$BASE_DIR/out_put"
name=ESA_Tianbao_age
submit_vars_file=""
ins=""
dry_run=0

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

while getopts ':n:s:i:th' opt; do
    case "$opt" in
        n) name=$OPTARG ;;
        s) submit_vars_file=$OPTARG ;;
        i) ins=$OPTARG ;;
        t) dry_run=1 ;;
        h) echo 'Usage: bash submit.sh [-n name] [-s settings.sh] [-i main.ins] [-t]'; exit 0 ;;
        :) fail "Option -$OPTARG requires a value." ;;
        \?) fail "Unknown option: -$OPTARG" ;;
    esac
done
shift "$((OPTIND - 1))"
(( $# == 0 )) || fail 'Unexpected positional arguments.'

if [[ -n "$submit_vars_file" ]]; then
    [[ -r "$submit_vars_file" ]] || fail "Cannot read settings: $submit_vars_file"
    source "$submit_vars_file"
fi
[[ -z "$ins" ]] || INSFILE=$ins
[[ "$NPROCESS" =~ ^[1-9][0-9]*$ ]] || fail 'NPROCESS must be a positive integer.'
[[ -n "$name" && -n "$OUTFILES" ]] || fail 'Job name and OUTFILES must not be empty.'

[[ -s "$INSFILE" ]] || fail "Missing or empty instruction file: $INSFILE"
[[ -s "$GRIDLIST" ]] || fail "Missing or empty gridlist: $GRIDLIST"
[[ -x "$EXE" ]] || fail "Executable not found or not executable: $EXE"
INSFILE=$(readlink -f "$INSFILE")
GRIDLIST=$(readlink -f "$GRIDLIST")
EXE=$(readlink -f "$EXE")
OUT_DIR=$(readlink -m "$OUT_DIR")
GRIDLIST_FILENAME=$(basename "$GRIDLIST")

# Read coordinate records, not newline characters. Keep optional site labels.
# Ignore blank lines, full-line comments and a conventional lon/lat header.
tmp_grid=$(mktemp)
trap 'rm -f -- "$tmp_grid"' EXIT
awk '
BEGIN { num="^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$" }
{
    sub(/^\357\273\277/, "")
    sub(/\r$/, "")
}
/^[[:space:]]*$/ || /^[[:space:]]*[#!]/ { next }
tolower($1)=="lon" && tolower($2)=="lat" { next }
{
    if (NF<2 || $1 !~ num || $2 !~ num || $1+0 < -180 || $1+0 > 180 || $2+0 < -90 || $2+0 > 90) {
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
if (( NPROCESS > Ngridcells )); then
    echo "INFO: Reducing NPROCESS from $NPROCESS to $Ngridcells to avoid empty ranks."
    NPROCESS=$Ngridcells
fi

# Never reuse output folders that may contain old or running results.
if [[ -e "$OUT_DIR" || -L "$OUT_DIR" ]]; then
    [[ -d "$OUT_DIR" && ! -L "$OUT_DIR" ]] || fail "Output path is not a regular directory: $OUT_DIR"
    [[ -z "$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]] ||
        fail "Output directory is not empty: $OUT_DIR. Preserve/rename it before a new run."
fi

printf '\nProject: %s\nInstruction file: %s\nGridlist: %s\nOutput: %s\n' \
    "$BASE_DIR" "$INSFILE" "$GRIDLIST" "$OUT_DIR"
printf 'Gridcells: %s\nMPI processes: %s\n' "$Ngridcells" "$NPROCESS"
q=$((Ngridcells / NPROCESS))
r=$((Ngridcells % NPROCESS))
for ((i=1; i<=NPROCESS; i++)); do
    count=$q
    if (( i <= r )); then count=$((count + 1)); fi
    printf '  run%d: %d gridcell(s)\n' "$i" "$count"
done
if (( dry_run )); then
    echo 'CHECK ONLY: No output folders created and no jobs submitted.'
    exit 0
fi
command -v sbatch >/dev/null || fail 'sbatch is not available in this shell.'

# An atomic directory creation also prevents simultaneous reuse of this path.
mkdir -p "$OUT_DIR"
mkdir "$OUT_DIR/.submission_lock" || fail 'Another submission is using this output directory.'
cp "$tmp_grid" "$OUT_DIR/gridlist_used.txt"
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

# Preserve the optional node-local setup hook if supplied with the project.
if [[ -f "$BASE_DIR/setup_local.sh" ]]; then
    cp "$BASE_DIR/setup_local.sh" ./setup_local.sh
fi

{
    printf '#!/bin/bash\n#SBATCH --ntasks=%d\n#SBATCH --cpus-per-task=1\n' "$NPROCESS"
    printf '#SBATCH --time=%s\n#SBATCH --exclusive\n#SBATCH --output=slurm-%%j.log\n' "$WALLTIME"
    printf 'NPROCESS=%q\nEXE=%q\nINSFILE=%q\nINPUT_MODULE=%q\nOUT_DIR=%q\n' \
        "$NPROCESS" "$EXE" "$INSFILE" "$INPUT_MODULE" "$OUT_DIR"
    cat <<'JOB'
set -eo pipefail
module purge
module load gompi/2025a netCDF/4.9.3
set -u
cd "$OUT_DIR"
command -v mpirun >/dev/null || { echo 'ERROR: mpirun was not found.' >&2; exit 1; }
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ -f ./setup_local.sh ]]; then
    srun -n "$SLURM_NNODES" -N "$SLURM_NNODES" --ntasks-per-node=1 bash ./setup_local.sh
fi
if [[ -n "${SNIC_TMP:-}" ]]; then
    for ((i=1; i<=NPROCESS; i++)); do
        ln -s "$SNIC_TMP" "run$i/local"
    done
else
    echo 'INFO: SNIC_TMP is unset; no optional local-storage links were created.'
fi

mpirun -np "$NPROCESS" --bind-to core "$EXE" \
    -parallel -input "$INPUT_MODULE" "$INSFILE"
JOB
} > guess.cmd

{
    printf '#!/bin/bash\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=1\n'
    printf '#SBATCH --time=%s\n#SBATCH --output=slurm-%%j.log\n' "$WALLTIME_APPEND"
    printf 'NPROCESS=%q\nOUTFILES=%q\nOUT_DIR=%q\n' "$NPROCESS" "$OUTFILES" "$OUT_DIR"
    cat <<'APPEND'
set -euo pipefail
cd "$OUT_DIR"
shopt -s nullglob
read -r -a patterns <<< "$OUTFILES"
declare -A names=()

# Use the union of file names, not only the files produced by run1.
for ((i=1; i<=NPROCESS; i++)); do
    [[ -s "run$i/guess.log" ]] || { echo "ERROR: Missing run$i/guess.log" >&2; exit 1; }
    found=0
    for pattern in "${patterns[@]}"; do
        for path in "run$i"/$pattern; do
            [[ -s "$path" && -f "$path" ]] || continue
            names["${path##*/}"]=1
            found=1
        done
    done
    (( found )) || { echo "ERROR: No nonempty output files in run$i" >&2; exit 1; }
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
    # Merge single-header tables or numeric headerless records.
    # Stop on incompatible headers instead of silently combining columns.
    awk '
    function norm(s) {
        sub(/\r$/, "", s); gsub(/[ \t]+/, " ", s)
        sub(/^ /, "", s); sub(/ $/, "", s); return s
    }
    BEGIN { num="^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$" }
    FNR==1 { first_record=1 }
    first_record && NF==0 { next }
    first_record {
        first_record=0
        current_has_header=($1 !~ num)
        if (!initialized) {
            initialized=1; has_header=current_has_header; header=norm($0)
            print; next
        }
        if (current_has_header!=has_header || (has_header && norm($0)!=header)) {
            printf "ERROR: Output headers differ in %s\n", FILENAME > "/dev/stderr"
            exit 1
        }
        if (!has_header) print
        next
    }
    { print }
    ' "${sources[@]}" > "$file.tmp" || { rm -f -- "$file.tmp"; exit 1; }
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

{
    printf '#!/bin/bash\ncd %q || exit 1\n' "$OUT_DIR"
    printf 'for ((i=1; i<=%d; i++)); do\n' "$NPROCESS"
    cat <<'PROGRESS'
    echo "===== run$i ====="
    if [[ -f "run$i/guess.log" ]]; then
        tail -n 8 "run$i/guess.log"
    else
        echo 'No log yet.'
    fi
done
PROGRESS
} > progress.sh

bash -n guess.cmd
bash -n append.cmd
bash -n progress.sh
account_args=()
if [[ -n "$COMPUTEACCOUNT" ]]; then account_args=(-A "$COMPUTEACCOUNT"); fi

job_response=$(sbatch --parsable "${account_args[@]}" --chdir="$OUT_DIR" -J "$name" guess.cmd)
job_id=${job_response%%;*}
[[ "$job_id" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: $job_response"
printf 'LPJ-GUESS job: %s\n' "$job_id"
printf 'model_job=%s\n' "$job_id" > job_ids.txt

if append_response=$(sbatch --parsable "${account_args[@]}" --chdir="$OUT_DIR" \
    --dependency="afterok:$job_id" -J "ap_${name#bm_}" append.cmd); then
    append_id=${append_response%%;*}
    [[ "$append_id" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch response: $append_response"
    printf 'Merge job: %s\n' "$append_id"
    printf 'merge_job=%s\n' "$append_id" >> job_ids.txt
else
    echo "ERROR: Model job $job_id was submitted, but merge-job submission failed." >&2
    echo 'Do not rerun submit.sh; submit append.cmd with the same dependency instead.' >&2
    exit 1
fi
printf 'Output directory: %s\n' "$OUT_DIR"
printf 'Progress command: bash %q/progress.sh\n' "$OUT_DIR"
