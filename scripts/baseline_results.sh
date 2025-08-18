#!/bin/bash

for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4 LaMP-5 LaMP-7 LongLaMP-2 LongLaMP-3 LongLaMP-4; do
    for baseline in bm25 contriever rank_gpt icr; do
        for llm in phi-4-mini-instruct llama-3-8b-instruct; do
            echo ${task}_${baseline}_${llm}
            python src/process.py baseline_results $task $baseline $llm
        done
    done
done
