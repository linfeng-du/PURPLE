#!/bin/bash

for task in LaMP-1 LaMP-2 LaMP-3 LaMP-4 LaMP-5 LaMP-7 LongLaMP-2 LongLaMP-3 LongLaMP-4; do
    for version in $1; do
        for llm in phi-4-mini-instruct llama-3-8b-instruct; do
            echo ${task}_${version}_${llm}
            python src/process.py bandit_pr_results $task $version $llm
        done
    done
done
