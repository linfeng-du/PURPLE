import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding


class ProfileScoreModel(nn.Module):

    def __init__(self, bert_encoder: str, hidden_size: int) -> None:
        """Initialize the ProfileScoreModel.

        Args:
            bert_encoder (str): Hugging Face identifier for a BERT-like encoder.
            hidden_size (int): Decoder hidden dimension size.
        """
        super().__init__()
        decoder_input_size = 2 * bert_encoder.config.hidden_size

        self.bert_encoder = AutoModel.from_pretrained(bert_encoder)
        self.norm = nn.LayerNorm(decoder_input_size)

        self.mlp_decoder = nn.Sequential(
            nn.Linear(decoder_input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: BatchEncoding,
        profile_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute profile likelihoods conditioned on the given query.

        Args:
            query_inputs (BatchEncoding): Query inputs prepared for the encoder.
            corpus_inputs (BatchEncoding): Corpus inputs prepared for the encoder.
            profile_mask (torch.Tensor): Validity of profiles. Shape (batch_size, n_profiles)

        Returns:
            likelihoods (torch.Tensor):
                Profile Likelihoods given the query. Shape (batch_size, n_profiles)
        """
        batch_size, n_profiles = profile_mask.size()

        query_embedding = self._compute_sentence_embedding(query_inputs)
        corpus_embeddings = self._compute_sentence_embedding(corpus_inputs)

        query_embedding = query_embedding.unsqueeze(dim=1).expand(-1, n_profiles, -1)
        corpus_embeddings = corpus_embeddings.view(batch_size, n_profiles, -1)
        query_corpus = torch.cat((query_embedding, corpus_embeddings), dim=-1)

        likelihoods = self.mlp_decoder(query_corpus)
        likelihoods = likelihoods.squeeze(dim=-1).masked_fill(~profile_mask, 0.)
        return likelihoods

    def _compute_sentence_embedding(self, sentence_inputs):
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

        token_embeddings = token_embeddings.masked_fill(attention_mask == 0, 0.)
        sentence_embedding = token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        return sentence_embedding
