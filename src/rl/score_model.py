import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding


class ProfileScoreModel(nn.Module):

    def __init__(
        self,
        bert_encoder: str,
        n_candidates: int,
        n_heads: int,
        hidden_size: int
    ) -> None:
        """Initialize the ProfileScoreModel.

        Args:
            bert_encoder (str): Hugging Face identifier for a BERT-like encoder.
            hidden_size (int): Decoder hidden dimension size.
        """
        super().__init__()
        self.n_candidates = n_candidates

        self.bert_encoder = AutoModel.from_pretrained(bert_encoder)
        encoder_hidden_size = self.bert_encoder.config.hidden_size

        self.mixer_mlp = nn.Sequential(
            nn.Linear(2 * encoder_hidden_size, encoder_hidden_size),
            nn.ReLU()
        )
        self.mixer_norm = nn.LayerNorm(encoder_hidden_size)

        self.attn = nn.MultiheadAttention(encoder_hidden_size, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(encoder_hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(encoder_hidden_size, 4 * encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(4 * encoder_hidden_size, encoder_hidden_size)
        )
        self.ffn_norm = nn.LayerNorm(encoder_hidden_size)

        self.mlp_decoder = nn.Sequential(
            nn.Linear(encoder_hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        for param in self.bert_encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        query_inputs: BatchEncoding,
        all_corpus_inputs: list[BatchEncoding],
        profile_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute candidate profile likelihoods conditioned on the given query.

        Args:
            query_inputs (BatchEncoding): Query inputs prepared for the encoder.
            corpus_inputs (BatchEncoding): Corpus inputs prepared for the encoder.
            profile_mask (torch.Tensor): Validity of profiles. Shape (batch_size, n_profiles)

        Returns:
            profile_likelihoods (torch.Tensor):
                Profile Likelihoods given the query. Shape (batch_size, n_profiles)
        """
        batch_size, n_profiles = profile_mask.size()
        n_candidates = min(self.n_candidates, n_profiles)
        batch_indices = torch.arange(batch_size).unsqueeze(dim=1)

        # Compute query and corpus embeddings
        query_embedding = self._compute_sentence_embedding(query_inputs)

        all_corpus_embeddings = []

        for corpus_inputs in all_corpus_inputs:
            corpus_embeddings = self._compute_sentence_embedding(corpus_inputs)
            all_corpus_embeddings.append(corpus_embeddings)

        corpus_embeddings = torch.cat(all_corpus_embeddings, dim=0)

        query_embedding = query_embedding.unsqueeze(dim=1)
        corpus_embeddings = corpus_embeddings.view(batch_size, n_profiles, -1)

        # Select candidate profiles
        scores = (query_embedding @ corpus_embeddings.transpose(dim0=-1, dim1=-2)).squeeze(dim=1)
        _, candidate_indices = scores.topk(n_candidates, dim=-1)
        candidate_mask = profile_mask.gather(dim=1, index=candidate_indices)

        # Mix query and candidate embeddings
        query_embedding = query_embedding.expand(-1, n_candidates, -1)
        candidate_embeddings = corpus_embeddings[batch_indices, candidate_indices, :]

        mixed_embeddings = torch.cat((query_embedding, candidate_embeddings), dim=-1)
        mixed_embeddings = self.mixer_norm(self.mixer_mlp(mixed_embeddings))

        # Model candidate profile dependencies
        attn_out, _ = self.attn(
            mixed_embeddings,
            mixed_embeddings,
            mixed_embeddings,
            key_padding_mask=~candidate_mask
        )
        mixed_embeddings = self.attn_norm(mixed_embeddings + attn_out)

        ffn_out = self.ffn(mixed_embeddings)
        mixed_embeddings = self.ffn_norm(mixed_embeddings + ffn_out)

        # Compute candidate profile likelihoods
        candidate_likelihoods = self.mlp_decoder(mixed_embeddings)
        candidate_likelihoods = candidate_likelihoods.squeeze(dim=-1)
        candidate_likelihoods = candidate_likelihoods.masked_fill(~candidate_mask, value=0.)

        return candidate_likelihoods, candidate_mask, candidate_indices

    def _compute_sentence_embedding(self, sentence_inputs: BatchEncoding):
        """Compute sentence embedding by mean pooling.

        Args:
            sentence_inputs (BatchEncoding): Sentence inputs prepared for the encoder.

        Returns:
            sentence_embedding (torch.Tensor):
                Averaged token embeddings. Shape (n_sentences, hidden_size)
        """
        sentence_outputs = self.bert_encoder(**sentence_inputs)

        token_embeddings = sentence_outputs.last_hidden_state
        attention_mask = sentence_inputs['attention_mask'].unsqueeze(dim=-1)

        token_embeddings.masked_fill_(attention_mask == 0, value=0.)
        sentence_embedding = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)

        return sentence_embedding
