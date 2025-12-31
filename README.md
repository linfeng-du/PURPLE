## Prepare datasets and pre-trained models

Download the LaMP and LongLaMP datasets, pre-trained models, and evaluation metrics by running:

```bash
bash scripts/download.sh
```

Correct the malformed example in `LaMP-2/dev_questions.json` by running:

```bash
python <<'EOF'
import json

with open("data/LaMP-2/dev_questions.json", "r") as f:
    examples = json.load(f)

INPUT_110 = (
    "Which tag does this movie relate to among the following tags? "
    "Just answer with the tag name without further explanation. "
    "tags: [sci-fi, based on a book, comedy, action, twist ending, "
    "dystopia, dark comedy, classic, psychology, fantasy, romance, "
    "thought-provoking, social commentary, violence, true story] "
    "description: Overwhelmed by her suffocating schedule, "
    "touring European princess Ann takes off for a night while in Rome. "
    "When a sedative she took from her doctor kicks in, however, "
    "she falls asleep on a park bench and is found by an American reporter, "
    "Joe Bradley, who takes her back to his apartment for safety. "
    "At work the next morning, Joe finds out Ann's regal identity and "
    "bets his editor he can get exclusive interview with her, "
    "but romance soon gets in the way."
)

for example in examples:
    if example["id"] == "110":
        example["input"] = INPUT_110

with open("data/LaMP-2/dev_questions.json", "w") as f:
    json.dump(examples, f)
EOF
```

## Environment
Setup environment for PURPLE by running:

```bash
virtualenv --no-download purple
source purple/bin/activate
pip install -r requirements.txt
```

Setup environment for `vLLM` by running:

```bash
virtualenv --no-download vllm
source vllm/bin/activate
pip install vllm flashinfer-python
```

## Preprocess datasets

Retrieve candidate records for each user, then tokenize the dataset.
Supported tasks are `LaMP-{1,2,3,4,5,7}` and `LongLaMP-{2,3,4}`.

Preprocess LaMP and LongLaMP datasets by running:

```bash
python src/preprocess.py --task=${TASK} --candidate_retriever=contriever --num_candidates=20
```

## Training
For experiments with `phi4-mini-instruct` and `llama3-8b-instruct`, the LLM can run on a single H100 GPU.
For experiments with `llama3-70b-instruct` and `qwen3next-80b-instruct`, serve the LLM on 4 H100 GPUs and provide the authority to the training script.
Evaluation on the test split runs after training.

Start training by running:

```bash
vllm serve \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor_parallel_size 4 \
    --host 0.0.0.0 \
    --port 8000
python src/train.py task=${TASK} llm=llama3-70b-instruct llm.authority=localhost:8000
```
