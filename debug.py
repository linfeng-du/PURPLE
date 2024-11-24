from argparse import Namespace

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from src.LaMP import (
    LaMPDataset,
    LaMPCollator,
    create_retrieval_prompt_generator,
    RetrieverTrainingDataset,
    RetrieverTrainingCollator,
    create_query_corpus_generator
)
from src.models import (
    RetrieverModel,
    Reinforce,
    create_reward
)


def train_retriever(cfg):
    # Prepare retrieval data
    train_loader = DataLoader(
        RetrieverTrainingDataset(
            f'./dataset/{cfg.task}/train_questions.json',
            f'./dataset/{cfg.task}/train_outputs.json',
            create_query_corpus_generator(cfg.task)
        ),
        batch_size=cfg.ret_cfg.num_sampled_questions,
        shuffle=True,
        collate_fn=RetrieverTrainingCollator(
            AutoTokenizer.from_pretrained(cfg.ret_cfg.retriever_model),
            cfg.ret_cfg.max_corpus_size,
            cfg.ret_cfg.max_query_length,
            cfg.ret_cfg.max_document_length
        )
    )

    # Prepare generation
    tokenizer = AutoTokenizer.from_pretrained(cfg.gen_cfg.generation_model)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    prompt_generator = create_retrieval_prompt_generator(
        cfg.task,
        'first_k',
        cfg.ret_cfg.num_retrieve,
        tokenizer,
        cfg.ret_cfg.max_prompt_length,
    )

    # Prepare models
    ret_model = RetrieverModel(AutoModel.from_pretrained(cfg.ret_cfg.retriever_model))
    gen_model = AutoModelForCausalLM.from_pretrained(cfg.gen_cfg.generation_model)
    ret_model.to(cfg.device)
    gen_model.to(cfg.device)

    reinforce = Reinforce(cfg.ret_cfg.num_sampled_retrievals)
    reward_fn = create_reward(cfg.task, tokenizer)

    optimizer = torch.optim.Adam(ret_model.parameters(), lr=cfg.ret_cfg.lr)

    # Training loop
    for batch in tqdm(train_loader):
        query = batch['query'].to(cfg.device)
        corpus = batch['corpus'].to(cfg.device)
        corpus_mask = batch['corpus_mask'].to(cfg.device)

        likelihoods = ret_model(query, corpus, corpus_mask)
        sample_idxs, log_prob = reinforce.sample(
            likelihoods,
            corpus_mask,
            cfg.ret_cfg.num_retrieve,
            cfg.ret_cfg.epsilon,
        )

        # Prepare generation data
        loader = DataLoader(
            LaMPDataset.from_batch_sample_indices(
                batch,
                sample_idxs,
                prompt_generator 
            ),
            batch_size=cfg.gen_cfg.batch_size,
            collate_fn=LaMPCollator(tokenizer, cfg.ret_cfg.max_prompt_length)
        )

        all_reward = []
        for batch in loader:
            batch = batch.to(cfg.device)
            outputs = gen_model.generate(
                **batch,
                max_new_tokens=cfg.gen_cfg.max_generation_length,
                num_beams=cfg.gen_cfg.num_beams,
                pad_token_id=tokenizer.eos_token_id
            )
            outputs = outputs[:, batch['input_ids'].size(dim=1):]
            reward = reward_fn(outputs, batch['labels'])
            all_reward.append(reward)

        reward = torch.cat(all_reward, dim=0).view_as(log_prob)

        loss = reinforce.compute_loss(log_prob, reward)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ret_model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()        


if __name__ == '__main__':
    ret_cfg = Namespace(
        retriever_model='facebook/contriever',

        max_corpus_size=50,
        max_query_length=8,
        max_document_length=18,

        num_sampled_questions=2,
        num_sampled_retrievals=20,
        num_retrieve=12,
        epsilon=0.1,
        lr=1e-5,

        max_prompt_length=512
    )
    gen_cfg = Namespace(
        generation_model='meta-llama/Llama-3.2-1B-Instruct',

        batch_size=16,
        max_generation_length=5,
        num_beams=5
    )
    cfg = Namespace(
        task='LaMP-7',
        ret_cfg=ret_cfg,
        gen_cfg=gen_cfg,
        device='cuda:0'
    )
    train_retriever(cfg)
