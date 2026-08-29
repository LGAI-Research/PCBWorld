#!/usr/bin/env bash
# Generate 1L 10net x 2pin datasets at grid sizes 10/30/100/300, sequentially.
# Each grid: 10K train (seed=0) + 128 test (seed=1) + 128 val (seed=2),
# plus combined_v2 dir and split JSON.
#
# Submit (from the repo root; add your site's --partition/--qos/--account):
#   sbatch tools/datagen/synthetic_generator/make_grid_datasets.sbatch

#SBATCH --job-name=mk_grid_datasets
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail

# Repo root: $REPO_DIR override > sbatch submit dir > this script's location
# (tools/datagen/synthetic_generator/ -> three levels up). Under sbatch the script
# itself runs from the Slurm spool copy, hence the SLURM_SUBMIT_DIR step.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}}"
CONDA_SH="${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"

if [[ ! -f "$REPO_DIR/tools/datagen/synthetic_generator/make_grid_dataset.sh" ]]; then
    echo "ERROR: REPO_DIR=$REPO_DIR is not the repo root (set REPO_DIR=... explicitly)" >&2
    exit 1
fi

cd "$REPO_DIR"
mkdir -p logs

source "$CONDA_SH"
conda activate cadagent

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-no_slurm}"
date

GRIDS=(10 30 100 300)
for g in "${GRIDS[@]}"; do
    echo ""
    echo "########################################"
    echo "# grid=$g  ($(date +%H:%M:%S))"
    echo "########################################"
    bash tools/datagen/synthetic_generator/make_grid_dataset.sh "$g"
done

echo ""
echo "All grids done at $(date)."
