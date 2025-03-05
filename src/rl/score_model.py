import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding


class ProfileScoreModel(nn.Module):

    def __init__(self, bert_encoder: str, n_candidates: int, hidden_size: int) -> None:
        """Initialize the ProfileScoreModel.

        Args:
            bert_encoder (str): Hugging Face identifier for a BERT-like encoder.
            hidden_size (int): Decoder hidden dimension size.
        """
        super().__init__()
        self.n_candidates = n_candidates

        self.bert_encoder = AutoModel.from_pretrained(bert_encoder)

        decoder_input_size = 2 * self.bert_encoder.config.hidden_size
        self.norm = nn.LayerNorm(decoder_input_size)

        self.mlp_decoder = nn.Sequential(
            nn.Linear(decoder_input_size, hidden_size),
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

        # Compute candidate profile likelihoods
        query_embedding = query_embedding.expand(-1, n_candidates, -1)
        candidate_embeddings = corpus_embeddings[batch_indices, candidate_indices, :]
        query_candidate_embeddings = torch.cat((query_embedding, candidate_embeddings), dim=-1)

        candidate_likelihoods = self.mlp_decoder(query_candidate_embeddings)
        candidate_likelihoods = candidate_likelihoods.squeeze(dim=-1)

        candidate_mask = profile_mask.gather(dim=1, index=candidate_indices)
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
