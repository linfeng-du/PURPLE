#!/bin/bash

sbatch_time='24:00:00'
tasks="LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,\
LongLaMP-2,LongLaMP-3,LongLaMP-4"
resume='false'
llms='phi4-mini-instruct,llama3-8b-instruct'
authority='null'

LONGOPTIONS='time:,tasks:,resume,llms:,authority:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) sbatch_time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --resume) resume='true'; shift 1 ;;
    --llms) llms="$2"; shift 2 ;;
    --authority) authority="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

readarray -td ',' tasks <<< "${tasks}"
readarray -td ',' llms <<< "${llms}"

for llm in "${llms[@]}"; do
  for task in "${tasks[@]}"; do
    job_name="${llm}/purple/${task}"
    log_dir="outputs/slurm/${job_name}"

    wrap_cmds=(
      'source ~/.bashrc;'
      'activate purple;'
      'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
      'python src/train.py'
      "task=${task}"
      "resume=${resume}"
      "llm=${llm}"
      "llm.authority=${authority}"
    )
    wrap_cmd="${wrap_cmds[*]}"

    mkdir -p "${log_dir}"
    sbatch \
      --job-name="${job_name}" \
      --time="${sbatch_time}" \
      --gpus-per-node=1 \
      --output="${log_dir}/%j.out" \
      --error="${log_dir}/%j.err" \
      --wrap="${wrap_cmd}"
  done
done
