import hydra
from omegaconf import DictConfig

from purple import create_pretokenize_fn, load_retrieved_lamp_dataset


@hydra.main(config_path="../conf", config_name="purple", version_base=None)
def preprocess(cfg: DictConfig) -> None:
    train_split = "train"
    train_dataset = load_retrieved_lamp_dataset(
        cfg.task, train_split, cfg.candidate_retriever, cfg.num_candidates
    )
    train_dataset.map(
        create_pretokenize_fn(**cfg.create_pretokenize_fn),
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )

    test_split = "dev" if cfg.task.startswith("lamp") else "test"
    test_dataset = load_retrieved_lamp_dataset(
        cfg.task, test_split, cfg.candidate_retriever, cfg.num_candidates
    )
    test_dataset.map(
        create_pretokenize_fn(**cfg.create_pretokenize_fn),
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )


if __name__ == "__main__":
    preprocess()
