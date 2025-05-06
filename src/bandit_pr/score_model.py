import os
import json
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding


logger = logging.getLogger(__name__)


class ScoreModel(nn.Module):

    def __init__(self, encoder_model: str, num_candidates: int, num_heads: int, decoder_hidden_size: int) -> None:
        """Initializes ScoreModel.
        The model ranks candidate profiles before computing their likelihoods.
        """
        super().__init__()
        self.encoder_model = encoder_model
        self.num_candidates = num_candidates
        self.num_heads = num_heads
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder = AutoModel.from_pretrained(self.encoder_model)
        self.encoder_hidden_size = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.mixer_mlp = nn.Sequential(
            nn.Linear(2 * self.encoder_hidden_size, self.encoder_hidden_size),
            nn.ReLU()
        )
        self.mixer_norm = nn.LayerNorm(self.encoder_hidden_size)

        self.attn = nn.MultiheadAttention(self.encoder_hidden_size, self.num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(self.encoder_hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, 4 * self.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(4 * self.encoder_hidden_size, self.encoder_hidden_size)
        )
        self.ffn_norm = nn.LayerNorm(self.encoder_hidden_size)

        self.mlp_decoder = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.decoder_hidden_size, 1),
            nn.Sigmoid()
        )

    @classmethod
    def from_pretrained(cls, model_name: str) -> 'ScoreModel':
        """Loads pretrained ScoreModel."""
        config_path = f'./models/{model_name}/config.json'
        state_dict_path = f'./models/{model_name}/model.bin'

        with open(config_path, 'r') as file:
            config = json.load(file)

        model = cls(**config)
        state_dict = torch.load(state_dict_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model

    def save_pretrained(self, model_name: str) -> None:
        """Saves pretrained ScoreModel."""
        os.makedirs(f'./models/{model_name}', exist_ok=True)
        config_path = f'./models/{model_name}/config.json'
        state_dict_path = f'./models/{model_name}/model.bin'

        with open(config_path, 'w') as file:
            config = {
                'encoder_model': self.encoder_model,
                'num_candidates': self.num_candidates,
                'num_heads': self.num_heads,
                'decoder_hidden_size': self.decoder_hidden_size
            }
            json.dump(config, file, indent=2)

        state_dict = OrderedDict({
            key: value.cpu()
            for key, value in self.state_dict().items()
            if not key.startswith('encoder.')}
        )
        torch.save(state_dict, state_dict_path)

    def forward(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[BatchEncoding],
        profile_mask: torch.Tensor
    ) -> torch.Tensor:
        """Computes candidate profile likelihoods conditioned on the given query.
        Candidate profiles are selected based on BERT score.
        """
        batch_size, num_profiles = profile_mask.size()
        batch_indices = torch.arange(batch_size).unsqueeze(dim=1)
        num_candidates = min(self.num_candidates, num_profiles)

        # Computes query and corpus embeddings
        query_embedding = self._compute_sentence_embedding(query_inputs)
        corpus_embeddings = []

        for document_inputs in corpus_inputs:
            document_embeddings = self._compute_sentence_embedding(document_inputs)
            corpus_embeddings.append(document_embeddings)

        query_embedding = query_embedding.unsqueeze(dim=1)
        corpus_embeddings = torch.sparse_coo_tensor(
            indices=profile_mask.nonzero().T,
            values=torch.cat(corpus_embeddings, dim=0),
            size=(batch_size, num_profiles, self.encoder_hidden_size)
        ).to_dense()

        # Selects candidate profiles
        scores = (query_embedding @ corpus_embeddings.transpose(dim0=1, dim1=2)).squeeze(dim=1)
        candidate_scores, candidate_indices = scores.topk(num_candidates, dim=1)

        # Indexes candidate embeddings and mask
        candidate_embeddings = corpus_embeddings[batch_indices, candidate_indices, :]
        candidate_mask = profile_mask.gather(dim=1, index=candidate_indices)

        # Mixes query and candidate embeddings
        query_embedding = query_embedding.expand(-1, num_candidates, -1)
        mixed_embeddings = torch.cat((query_embedding, candidate_embeddings), dim=2)
        mixer_out = self.mixer_norm(self.mixer_mlp(mixed_embeddings))

        # Models candidate profile dependencies
        attn_out, _ = self.attn(mixer_out, mixer_out, mixer_out, key_padding_mask=~candidate_mask)
        attn_out = self.attn_norm(mixer_out + attn_out)

        ffn_out = self.ffn(attn_out)
        ffn_out = self.ffn_norm(attn_out + ffn_out)

        # Computes candidate profile likelihoods
        candidate_likelihoods = self.mlp_decoder(ffn_out).squeeze(dim=2)
        candidate_likelihoods = candidate_likelihoods.masked_fill(~candidate_mask, value=0.)

        return candidate_likelihoods, candidate_mask, candidate_indices

    def _compute_sentence_embedding(self, sentence_inputs: BatchEncoding) -> torch.Tensor:
        """Computes sentence embedding by mean pooling over token embeddings."""
        sentence_outputs = self.encoder(**sentence_inputs)
        token_embeddings = sentence_outputs.last_hidden_state
        attention_mask = sentence_inputs['attention_mask'].unsqueeze(dim=2)

        token_embeddings.masked_fill_(attention_mask == 0, value=0.)
        sentence_embedding = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        return sentence_embedding
