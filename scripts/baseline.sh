#!/bin/bash

sbatch_time='24:00:00'
tasks="LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,\
LongLaMP-2,LongLaMP-3,LongLaMP-4"
retrievers='bm25,contriever,icralm,replug,rank_gpt-llama3,rank_gpt-gpt5,icr'
llms='phi4-mini-instruct,llama3-8b-instruct'
authority='null'

LONGOPTIONS='time:,tasks:,retrievers:,llms:,authority:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) sbatch_time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --retrievers) retrievers="$2"; shift 2 ;;
    --llms) llms="$2"; shift 2 ;;
    --authority) authority="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -r -a tasks <<< "${tasks}"
IFS=',' read -r -a retrievers <<< "${retrievers}"
IFS=',' read -r -a llms <<< "${llms}"

for llm in "${llms[@]}"; do
  for retriever in "${retrievers[@]}"; do
    for task in "${tasks[@]}"; do
      job_name="${llm}/${retriever}/${task}"
      log_dir="outputs/slurm/${job_name}"

      wrap_cmds=(
        'source ~/.bashrc;'
        'activate purple;'
        'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
        'python src/baseline.py'
        "task=${task}"
        "retriever=${retriever}"
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
done
