#!/bin/bash

BASE_URL='https://ciir.cs.umass.edu/downloads/LaMP'

mkdir -p data/lamp1
wget -P data/lamp1 "${BASE_URL}/LaMP_1/train/train_questions.json"
wget -P data/lamp1 "${BASE_URL}/LaMP_1/train/train_outputs.json"
wget -P data/lamp1 "${BASE_URL}/LaMP_1/dev/dev_questions.json"
wget -P data/lamp1 "${BASE_URL}/LaMP_1/dev/dev_outputs.json"

mkdir -p data/lamp2
wget -P data/lamp2 "${BASE_URL}/LaMP_2/new/train/train_questions.json"
wget -P data/lamp2 "${BASE_URL}/LaMP_2/new/train/train_outputs.json"
wget -P data/lamp2 "${BASE_URL}/LaMP_2/new/dev/dev_questions.json"
wget -P data/lamp2 "${BASE_URL}/LaMP_2/new/dev/dev_outputs.json"

mkdir -p data/lamp3
wget -P data/lamp3 "${BASE_URL}/LaMP_3/train/train_questions.json"
wget -P data/lamp3 "${BASE_URL}/LaMP_3/train/train_outputs.json"
wget -P data/lamp3 "${BASE_URL}/LaMP_3/dev/dev_questions.json"
wget -P data/lamp3 "${BASE_URL}/LaMP_3/dev/dev_outputs.json"

mkdir -p data/lamp4
wget -P data/lamp4 "${BASE_URL}/LaMP_4/train/train_questions.json"
wget -P data/lamp4 "${BASE_URL}/LaMP_4/train/train_outputs.json"
wget -P data/lamp4 "${BASE_URL}/LaMP_4/dev/dev_questions.json"
wget -P data/lamp4 "${BASE_URL}/LaMP_4/dev/dev_outputs.json"

mkdir -p data/lamp5
wget -P data/lamp5 "${BASE_URL}/LaMP_5/train/train_questions.json"
wget -P data/lamp5 "${BASE_URL}/LaMP_5/train/train_outputs.json"
wget -P data/lamp5 "${BASE_URL}/LaMP_5/dev/dev_questions.json"
wget -P data/lamp5 "${BASE_URL}/LaMP_5/dev/dev_outputs.json"

mkdir -p data/lamp7
wget -P data/lamp7 "${BASE_URL}/LaMP_7/train/train_questions.json"
wget -P data/lamp7 "${BASE_URL}/LaMP_7/train/train_outputs.json"
wget -P data/lamp7 "${BASE_URL}/LaMP_7/dev/dev_questions.json"
wget -P data/lamp7 "${BASE_URL}/LaMP_7/dev/dev_outputs.json"

python <<'EOF'
from datasets import load_dataset

load_dataset("LongLaMP/LongLaMP", name="abstract_generation_user")
load_dataset("LongLaMP/LongLaMP", name="topic_writing_user")
load_dataset("LongLaMP/LongLaMP", name="product_review_user")
EOF

hf download facebook/contriever
hf download microsoft/Phi-4-mini-instruct
hf download meta-llama/Meta-Llama-3-8B-Instruct
hf download meta-llama/Meta-Llama-3-70B-Instruct

python <<'EOF'
from evaluate import load

load("accuracy")
load("f1")
load("mae")
load("mse")
load("rouge")
load("meteor")
EOF
