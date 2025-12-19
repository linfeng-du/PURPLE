#!/bin/bash

sbatch_time='6:00:00'
tasks="LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,\
LongLaMP-2,LongLaMP-3,LongLaMP-4"
candidate_retrievers='bm25,contriever'

LONGOPTIONS='time:,tasks:,candidate_retrievers:'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) sbatch_time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --candidate_retrievers) candidate_retrievers="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

readarray -td ',' tasks <<< "${tasks}"
readarray -td ',' candidate_retrievers <<< "${candidate_retrievers}"

for candidate_retriever in "${candidate_retrievers[@]}"; do
  for task in "${tasks[@]}"; do
    job_name="preprocess/${candidate_retriever}/${task}"
    log_dir="outputs/slurm/${job_name}"

    wrap_cmds=(
      'source ~/.bashrc;'
      'activate purple;'
      'python src/preprocess.py'
      "task=${task}"
      "candidate_retriever=${candidate_retriever}"
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
