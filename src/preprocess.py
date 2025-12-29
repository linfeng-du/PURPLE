import fire
from transformers import AutoTokenizer

from purple import create_preprocess_fn, load_retrieved_lamp_dataset


def preprocess(
    task: str,
    candidate_retriever: str,
    num_candidates: int
) -> None:
    train_split = "train"
    test_split = ("dev" if task.startswith("LaMP-") else "test")

    train_dataset = load_retrieved_lamp_dataset(
        task, train_split, candidate_retriever, num_candidates
    )
    test_dataset = load_retrieved_lamp_dataset(
        task, test_split, candidate_retriever, num_candidates
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    preprocess_fn = create_preprocess_fn(
        max_query_length=512,
        max_document_length=512,
        tokenizer=tokenizer
    )
    train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )

    # Re-create tokenizer to keep the `.map()` fingerprint deterministic
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    preprocess_fn = create_preprocess_fn(
        max_query_length=512,
        max_document_length=512,
        tokenizer=tokenizer
    )
    test_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=["query", "corpus"],
        num_proc=4
    )


if __name__ == "__main__":
    fire.Fire(preprocess)
