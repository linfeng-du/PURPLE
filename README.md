## Environment setup

Create a virtual environment for PURPLE.
Install packages from `requirements.txt` and `flash-attn`.

```bash
virtualenv --no-download purple
source purple/bin/activate
pip install -r requirements.txt flash-attn==2.8.3
```

Create a separate virtual environment for `vllm`.

```bash
virtualenv --no-download vllm
source vllm/bin/activate
pip install vllm==0.15.1
```

## Download resources

Download LaMP and LongLaMP datasets, pre-trained models, and evaluation metrics.

```bash
bash scripts/download.sh
```

Correct the malformed example in `data/lamp2/dev_questions.json`.

```bash
python <<'EOF'
import json
from pathlib import Path

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

file = Path("data/lamp2/dev_questions.json")
examples = json.loads(file.read_text())

for example in examples:
    if example["id"] == "110":
        example["input"] = INPUT_110

file.write_text(json.dumps(examples))
EOF
```

## Preprocess datasets

Retrieve candidate records for each user, then pre-tokenize the dataset.

Supported tasks: `lamp{1,2,3,4,5,7}` and `longlamp{2,3,4}`.

```bash
python src/preprocess.py task=${TASK}
```

## Training PURPLE

For experiments with `phi4-mini-instruct` and `llama3-8b-instruct`, training can be performed on a single H100 GPU.

```bash
python src/train.py task=${TASK} llm=${LLM}
```

For experiments with `llama3-70b-instruct`, serve the LLM on four H100 GPUs, then connect the training script to the inference endpoint.

```bash
vllm serve meta-llama/Meta-Llama-3-70B-Instruct --tensor-parallel-size 4
python src/train.py task=${TASK} llm=llama3-70b-instruct llm.host=${HOST}
```

Testing is performed after training.
