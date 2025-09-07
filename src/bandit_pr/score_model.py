import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding


class ScoreModel(nn.Module):

    def __init__(
        self,
        encoder_model: str,
        fuse_mode: str,
        num_layers: int,
        decoder_hidden_size: int
    ) -> None:
        super().__init__()
        self.encoder_model = encoder_model
        self.fuse_mode = fuse_mode
        self.num_layers = num_layers
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = AutoModel.from_pretrained(self.encoder_model)
        self.encoder_hidden_size = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.fuse_mode == 'concat_hidden':
            self.fuse_mlp = nn.Sequential(
                nn.Linear(2 * self.encoder_hidden_size, self.encoder_hidden_size),
                nn.ReLU()
            )
        elif self.fuse_mode == 'concat_token':
            pass
        elif self.fuse_mode == 'cross_attn':
            self.fuse_attn = nn.MultiheadAttention(
                self.encoder_hidden_size,
                self.encoder.config.num_attention_heads,
                batch_first=True
            )
        else:
            raise ValueError(f'Invalid fuse mode: {self.fuse_mode}')

        self.fuse_norm = nn.LayerNorm(self.encoder_hidden_size)
        self.doc_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.encoder_hidden_size,
                self.encoder.config.num_attention_heads,
                dim_feedforward=4 * self.encoder_hidden_size,
                batch_first=True
            ),
            num_layers,
            enable_nested_tensor=False
        )

        self.mlp_decoder = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_size, 1),
            nn.Sigmoid()
        )

    @classmethod
    def from_pretrained(cls, ckpt_dir: str) -> 'ScoreModel':
        config_path = f'{ckpt_dir}/config.json'
        state_dict_path = f'{ckpt_dir}/model.pt'

        with open(config_path, 'r') as file:
            config = json.load(file)

        model = cls(**config)
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model

    def save_pretrained(self, ckpt_dir: str) -> None:
        os.makedirs(f'{ckpt_dir}', exist_ok=True)
        config_path = f'{ckpt_dir}/config.json'
        state_dict_path = f'{ckpt_dir}/model.pt'

        with open(config_path, 'w') as file:
            config = {
                'encoder_model': self.encoder_model,
                'fuse_mode': self.fuse_mode,
                'num_layers': self.num_layers,
                'decoder_hidden_size': self.decoder_hidden_size
            }
            json.dump(config, file, indent=2)

        state_dict = OrderedDict({
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith('encoder.')}
        )
        torch.save(state_dict, state_dict_path)

    def forward(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        profile_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.fuse_mode == 'concat_hidden':
            fuse_embeds = self._fuse_concat_hidden(query_inputs, corpus_inputs, profile_mask)
        elif self.fuse_mode == 'concat_token':
            fuse_embeds, fuse_mask = self._fuse_concat_token(query_inputs, corpus_inputs, profile_mask)
        elif self.fuse_mode == 'cross_attn':
            fuse_embeds = self._fuse_cross_attention(query_inputs, corpus_inputs, profile_mask)

        # Model candidate profile dependencies
        fuse_embeds = self.fuse_norm(fuse_embeds)
        src_key_padding_mask = ~(fuse_mask if self.fuse_mode == 'concat_token' else profile_mask)
        transformer_out = self.doc_transformer(fuse_embeds, src_key_padding_mask=src_key_padding_mask)

        # Compute profile likelihoods
        likelihoods = self.mlp_decoder(transformer_out).squeeze(dim=2)

        if self.fuse_mode == 'concat_token':
            likelihoods = likelihoods[:, 1:]

        return likelihoods.masked_fill(~profile_mask, value=0.)

    def _fuse_concat_hidden(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        profile_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_profiles = profile_mask.shape

        query_embed = self._compute_sentence_embedding(query_inputs)
        corpus_embeds = [
            self._compute_sentence_embedding(document_inputs)
            for document_subbatches in corpus_inputs
            for document_inputs in document_subbatches
        ]

        query_embed = query_embed.unsqueeze(dim=1)
        corpus_embeds = torch.sparse_coo_tensor(
            indices=profile_mask.nonzero().T,
            values=torch.cat(corpus_embeds, dim=0),
            size=(batch_size, num_profiles, self.encoder_hidden_size)
        ).to_dense()

        query_embed = query_embed.expand(-1, num_profiles, -1)
        return self.fuse_mlp(torch.cat([query_embed, corpus_embeds], dim=2))

    def _fuse_concat_token(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        profile_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_profiles = profile_mask.shape

        query_embed = self._compute_sentence_embedding(query_inputs)
        corpus_embeds = [
            self._compute_sentence_embedding(document_inputs)
            for document_subbatches in corpus_inputs
            for document_inputs in document_subbatches
        ]
        corpus_embeds = torch.sparse_coo_tensor(
            indices=profile_mask.nonzero().T,
            values=torch.cat(corpus_embeds, dim=0),
            size=(batch_size, num_profiles, self.encoder_hidden_size)
        ).to_dense()

        fuse_embeds = torch.cat([query_embed.unsqueeze(dim=1), corpus_embeds], dim=1)
        fuse_mask = torch.cat([torch.ones_like(profile_mask[:, :1]), profile_mask], dim=1)
        return fuse_embeds, fuse_mask

    def _fuse_cross_attention(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        profile_mask: torch.Tensor
    ) -> torch.Tensor:
        query_mask = query_inputs['attention_mask'].bool()
        batch_size, num_profiles = profile_mask.shape

        query_token_embeds = self.encoder(**query_inputs).last_hidden_state
        fuse_embeds = []

        for index, document_subbatches in enumerate(corpus_inputs):
            for document_inputs in document_subbatches:
                num_documents = document_inputs['attention_mask'].shape[0]
                document_mask = document_inputs['attention_mask'].unsqueeze(dim=2)
                document_token_embeds = self.encoder(**document_inputs).last_hidden_state

                attn_out, _ = self.fuse_attn(
                    document_token_embeds,
                    query_token_embeds[index].expand(num_documents, -1, -1),
                    query_token_embeds[index].expand(num_documents, -1, -1),
                    key_padding_mask=~query_mask[index].expand(num_documents, -1)
                )
                attn_out.masked_fill_(document_mask == 0, value=0.)
                fuse_embed = attn_out.sum(dim=1) / document_mask.sum(dim=1)
                fuse_embeds.append(fuse_embed)

        return torch.sparse_coo_tensor(
            indices=profile_mask.nonzero().T,
            values=torch.cat(fuse_embeds, dim=0),
            size=(batch_size, num_profiles, self.encoder_hidden_size)
        ).to_dense()

    def _compute_sentence_embedding(self, sentence_inputs: BatchEncoding) -> torch.Tensor:
        attention_mask = sentence_inputs['attention_mask'].unsqueeze(dim=2)
        token_embeds = self.encoder(**sentence_inputs).last_hidden_state
        token_embeds.masked_fill_(attention_mask == 0, value=0.)
        return token_embeds.sum(dim=1) / attention_mask.sum(dim=1)
