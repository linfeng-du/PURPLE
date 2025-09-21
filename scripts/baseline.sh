#!/bin/bash

ARGS=$(getopt \
    --options "" \
    --long time:,tasks:,llms:,endpoint:,retrievers:,num_candidates:,rerankers:,num_rerank: \
    --name "$0" \
    -- "$@"
)
eval set -- "$ARGS"

while true; do
    case "$1" in
        --time) time="$2"; shift 2 ;;
        --tasks) tasks="$2"; shift 2 ;;
        --llms) llms="$2"; shift 2 ;;
        --endpoint) endpoint="$2"; shift 2 ;;
        --retrievers) retrievers="$2"; shift 2 ;;
        --num_candidates) num_candidates="$2"; shift 2 ;;
        --rerankers) rerankers="$2"; shift 2 ;;
        --num_rerank) num_rerank="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

: "${time:=24:00:00}"
: "${tasks:=LaMP-1,LaMP-2,LaMP-3,LaMP-4,LaMP-5,LaMP-7,LongLaMP-2,LongLaMP-3,LongLaMP-4}"
: "${llms:=phi-4-mini-instruct,llama-3-8b-instruct}"
: "${endpoint:=null}"
: "${retrievers:=contriever}"
: "${num_candidates:=20}"
: "${rerankers:=icr,rank_gpt-gpt5,rank_gpt-llama3,replug,icralm,contriever,bm25}"
: "${num_rerank:=5}"

IFS=',' read -ra tasks <<< "$tasks"
IFS=',' read -ra llms <<< "$llms"
IFS=',' read -ra retrievers <<< "$retrievers"
IFS=',' read -ra rerankers <<< "$rerankers"

for llm in ${llms[@]}; do
    for retriever in ${retrievers[@]}; do
        for reranker in ${rerankers[@]}; do
            for task in ${tasks[@]}; do
                exp_name="$llm/$retriever-$num_candidates/$reranker-$num_rerank/$task"
                mkdir -p "./logs/$exp_name"
                sbatch \
                    --job-name="$exp_name" \
                    --time="$time" \
                    --gres=gpu:h100:1 \
                    --mem=128G \
                    --output="./logs/$exp_name/%j.out" \
                    --error="./logs/$exp_name/%j.err" \
                    --wrap="$(
                        echo -n "source ~/.bashrc; "
                        echo -n "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
                        echo -n "activate bandit_ramp; "
                        echo -n "python src/baseline.py "
                        echo -n "exp_name=\"$exp_name\" "
                        echo -n "task=\"$task\" "
                        echo -n "llm=\"$llm\" "
                        echo -n "llm.endpoint=\"$endpoint\" "
                        echo -n "retriever=\"$retriever\" "
                        echo -n "num_candidates=\"$num_candidates\" "
                        echo -n "reranker=\"$reranker\" "
                        echo -n "num_rerank=\"$num_rerank\""
                    )"
            done
        done
    done
done
