#!/bin/bash

sbatch_time='24:00:00'
model='meta-llama/Meta-Llama-3-70B-Instruct'
port='8000'

LONGOPTIONS='time:,model:,port:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) sbatch_time="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

job_name="serve_vllm/${model}"
log_dir="outputs/slurm/${job_name}"

wrap_cmds=(
  'source ~/.bashrc;'
  'activate vllm;'
  "vllm serve ${model} --tensor_parallel_size 4 --host 0.0.0.0 --port ${port}"
)
wrap_cmd="${wrap_cmds[*]}"

mkdir -p "${log_dir}"
sbatch \
  --job-name="${job_name}" \
  --time="${sbatch_time}" \
  --gpus-per-node=4 \
  --output="${log_dir}/%j.out" \
  --error="${log_dir}/%j.err" \
  --wrap="${wrap_cmd}"
