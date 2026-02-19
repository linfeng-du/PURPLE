#!/bin/bash

time='00-06'
tasks='lamp1,lamp2,lamp3,lamp4,lamp5,lamp7,longlamp2,longlamp3,longlamp4'

LONGOPTIONS='time:,tasks:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -ra tasks <<< "${tasks}"

for task in "${tasks[@]}"; do
  job_name="preprocess/${task}"
  log_dir="outputs/slurm/${job_name}"

  wrap_cmds=(
    'source ~/.bashrc;'
    'activate purple;'
    'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
    'python src/preprocess.py'
    "task=${task}"
  )
  wrap_cmd="${wrap_cmds[*]}"

  mkdir -p "${log_dir}"
  sbatch \
    --job-name="${job_name}" \
    --time="${time}" \
    --gpus-per-node='1' \
    --output="${log_dir}/%j.out" \
    --error="${log_dir}/%j.err" \
    --wrap="${wrap_cmd}"
done
