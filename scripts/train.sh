#!/bin/bash

time='01-00'
tasks='lamp1,lamp2,lamp3,lamp4,lamp5,lamp7,longlamp2,longlamp3,longlamp4'
num_retrieve='5'
llms='phi4-mini-instruct,llama3-8b-instruct'

LONGOPTIONS='time:,tasks:,num_retrieve:,llms:,vllm_server_host:,resume'
TEMP=$(getopt --options '' --longoptions "${LONGOPTIONS}" --name "$0" -- "$@")
eval set -- "${TEMP}"

while true; do
  case "$1" in
    --time) time="$2"; shift 2 ;;
    --tasks) tasks="$2"; shift 2 ;;
    --num_retrieve) num_retrieve="$2"; shift 2 ;;
    --llms) llms="$2"; shift 2 ;;
    --vllm_server_host) vllm_server_host="$2"; shift 2 ;;
    --resume) resume="true"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

IFS=',' read -ra tasks <<< "${tasks}"
IFS=',' read -ra llms <<< "${llms}"

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
      "num_retrieve=${num_retrieve}"
      "llm=${llm}"
      ${vllm_server_host:+"llm.vllm_server_host=${vllm_server_host}"}
      ${resume:+"trainer_args.resume=true"}
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
done
