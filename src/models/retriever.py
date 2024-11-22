import torch
import torch.nn as nn


class RetrieverModel(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        hidden_size = 2 * self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, query_inputs, corpus_inputs, corpus_mask):
        batch_size, corpus_size = corpus_mask.size()

        query_embedding = self._compute_sentence_embeddings(query_inputs)
        corpus_embeddings = self._compute_sentence_embeddings(corpus_inputs)

        query_embedding = query_embedding.unsqueeze(dim=1).expand(-1, corpus_size, -1)
        corpus_embeddings = corpus_embeddings.view(batch_size, corpus_size, -1)
        query_corpus = torch.cat((query_embedding, corpus_embeddings), dim=-1)

        likelihoods = self.decoder(query_corpus)
        likelihoods = likelihoods.squeeze(dim=-1)
        likelihoods = likelihoods * corpus_mask
        return likelihoods

    def _compute_sentence_embeddings(self, sentence_inputs):
        sentence_outputs = self.encoder(**sentence_inputs)
        token_embeddings = sentence_outputs.last_hidden_state

        mask = sentence_inputs['attention_mask'].unsqueeze(dim=-1)
        token_embeddings = token_embeddings * mask
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)
        return sentence_embeddings
