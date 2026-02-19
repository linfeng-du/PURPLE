#!/bin/bash

time='01-00'
model='meta-llama/Meta-Llama-3-70B-Instruct'

TEMP=$(getopt --options '' --longoptions 'time:,model:' --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

job_name="vllm_serve/${model}"
log_dir="outputs/slurm/${job_name}"

wrap_cmds=(
  'source ~/.bashrc;'
  'activate vllm;'
  "vllm serve ${model} --tensor-parallel-size 4"
)
wrap_cmd="${wrap_cmds[*]}"

mkdir -p "${log_dir}"
sbatch \
  --job-name="${job_name}" \
  --time="${time}" \
  --gpus-per-node='4' \
  --output="${log_dir}/%j.out" \
  --error="${log_dir}/%j.err" \
  --wrap="${wrap_cmd}"
