from argparse import Namespace

from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.LaMP import (
    create_query_corpus_generator,
    Seq2SeqRetrieverTrainingDataset,
    CollatorForSeq2SeqRetrieverTraining
)
from src.models import LikelihoodModel
from src.reinforce import Reinforce


def train_retriever(configs):
    train_dataset = Seq2SeqRetrieverTrainingDataset(
        f'./dataset/{configs.task}/train_questions.json',
        f'./dataset/{configs.task}/train_outputs.json',
        create_query_corpus_generator(configs.task)
    )
    collate_fn = CollatorForSeq2SeqRetrieverTraining(
        configs.max_query_length,
        configs.max_corpus_length,
        configs.max_corpus_size,
        AutoTokenizer.from_pretrained(configs.model_name)
    )
    train_loader = DataLoader(
        train_dataset,
        configs.num_sampled_questions,
        shuffle=False,
        collate_fn=collate_fn
    )
    lik_model = LikelihoodModel(configs.model_name)
    reinforce = Reinforce(None)
    lik_model.to(configs.device)

    for batch in tqdm(train_loader):
        query = batch['query'].to(configs.device)
        corpus = batch['corpus'].to(configs.device)
        corpus_mask = batch['corpus_mask'].to(configs.device)
        likelihoods = lik_model(query, corpus, corpus_mask)
        reinforce.compute_loss(
            likelihoods,
            configs.num_sampled_retrievals,
            configs.num_retrieve,
            corpus_mask,
            configs.epsilon
        )


if __name__ == '__main__':
    configs = Namespace(
        task='LaMP-1',
        model_name='facebook/contriever',

        max_query_length=8,
        max_corpus_length=18,
        max_corpus_size=32,

        num_sampled_questions=32,
        num_sampled_retrievals=20,
        num_retrieve=12,
        epsilon=0.1,

        device='cuda:0'
    )
    train_retriever(configs)
