import hydra
from omegaconf import DictConfig

from purple import create_pretokenize_fn, load_or_create_retrieved_lamp_dataset


@hydra.main(config_path="../conf", config_name="purple", version_base=None)
def preprocess(cfg: DictConfig) -> None:
    train_dataset = load_or_create_retrieved_lamp_dataset(
        split="train", **cfg.load_or_create_retrieved_lamp_dataset
    )
    train_dataset.map(
        create_pretokenize_fn(**cfg.create_pretokenize_fn),
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )

    test_dataset = load_or_create_retrieved_lamp_dataset(
        split="dev" if cfg.task.startswith("lamp") else "test",
        **cfg.load_or_create_retrieved_lamp_dataset
    )
    test_dataset.map(
        create_pretokenize_fn(**cfg.create_pretokenize_fn),
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )


if __name__ == "__main__":
    preprocess()
