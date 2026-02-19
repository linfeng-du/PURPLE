import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BatchEncoding


class ScoreModel(nn.Module):
    def __init__(
        self,
        encoder_model: str,
        fuse_mode: str,
        num_transformer_layers: int,
        decoder_hidden_size: int
    ) -> None:
        super().__init__()
        self.encoder_model = encoder_model
        self.fuse_mode = fuse_mode
        self.num_transformer_layers = num_transformer_layers
        self.decoder_hidden_size = decoder_hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_model)
        self.encoder = AutoModel.from_pretrained(self.encoder_model)
        self.encoder_hidden_size = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.fuse_mode == "concat_hidden":
            self.fuse_linear = nn.Sequential(
                nn.Linear(
                    2 * self.encoder_hidden_size, self.encoder_hidden_size
                ),
                nn.ReLU()
            )
        elif self.fuse_mode == "cross_attn":
            self.fuse_attn = nn.MultiheadAttention(
                self.encoder_hidden_size,
                self.encoder.config.num_attention_heads,
                batch_first=True
            )

        self.fuse_norm = nn.LayerNorm(self.encoder_hidden_size)

        if self.num_transformer_layers > 0:
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    self.encoder_hidden_size,
                    self.encoder.config.num_attention_heads,
                    dim_feedforward=4 * self.encoder_hidden_size,
                    batch_first=True
                ),
                self.num_transformer_layers,
                enable_nested_tensor=False
            )

        self.decoder = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_size, 1),
            nn.Sigmoid()
        )

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ScoreModel":
        config = json.loads((Path(model_dir) / "config.json").read_text())
        model = cls(**config)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.load_pretrained(model_dir)
        return model

    def load_pretrained(self, model_dir: str) -> None:
        state_dict = torch.load(
            Path(model_dir) / "model.pt",
            map_location=next(self.parameters()).device,
            weights_only=True
        )
        self.load_state_dict(state_dict, strict=False)

    def save_pretrained(self, model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "encoder_model": self.encoder_model,
            "fuse_mode": self.fuse_mode,
            "num_transformer_layers": self.num_transformer_layers,
            "decoder_hidden_size": self.decoder_hidden_size
        }
        (model_dir / "config.json").write_text(json.dumps(config, indent=2))

        state_dict = OrderedDict({
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("encoder.")
        })
        torch.save(state_dict, model_dir / "model.pt")

    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        corpus: list[str],
        profile: list[dict[str, str]],
        num_retrieve: int
    ) -> list[dict[str, str]]:
        num_retrieve = min(num_retrieve, len(profile))

        query_inputs = (
            self.tokenizer(query, truncation=True, return_tensors="pt")
            .to(self.encoder.device)
        )
        corpus_inputs = [[
            (
                self.tokenizer(
                    corpus[i : i + 128],
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                .to(self.encoder.device)
            )
            for i in range(0, len(corpus), 128)
        ]]
        record_mask = (
            torch.ones(1, len(profile), dtype=torch.bool)
            .to(self.encoder.device)
        )

        likelihoods = self(query_inputs, corpus_inputs, record_mask)
        _, indices = likelihoods.topk(num_retrieve)
        return [profile[i] for i in indices.squeeze(dim=0)]

    def forward(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        record_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.fuse_mode == "concat_hidden":
            fuse_embs = self._fuse_concat_hidden(
                query_inputs, corpus_inputs, record_mask
            )
        elif self.fuse_mode == "concat_token":
            fuse_embs, fuse_mask = self._fuse_concat_token(
                query_inputs, corpus_inputs, record_mask
            )
        elif self.fuse_mode == "cross_attn":
            fuse_embs = self._fuse_cross_attention(
                query_inputs, corpus_inputs, record_mask
            )
        else:
            raise ValueError(f"Invalid fuse mode: {self.fuse_mode}")

        fuse_embs = self.fuse_norm(fuse_embs)

        if self.num_transformer_layers > 0:
            # Model record dependency
            src_key_padding_mask = ~(
                fuse_mask if self.fuse_mode == "concat_token" else record_mask
            )
            fuse_embs = self.transformer(
                fuse_embs, src_key_padding_mask=src_key_padding_mask
            )

        # Compute record likelihoods
        likelihoods = self.decoder(fuse_embs).squeeze(dim=-1)

        if self.fuse_mode == "concat_token":
            likelihoods = likelihoods[:, 1:]

        return likelihoods * record_mask

    def _fuse_concat_hidden(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        record_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_records = record_mask.shape

        query_emb = self._compute_sentence_embedding(query_inputs)
        corpus_embs = [
            self._compute_sentence_embedding(doc_inputs)
            for doc_batches in corpus_inputs
            for doc_inputs in doc_batches
        ]

        query_emb = query_emb.unsqueeze(dim=1).expand(-1, num_records, -1)
        corpus_embs = (
            torch.sparse_coo_tensor(
                record_mask.nonzero().T,
                torch.cat(corpus_embs),
                size=(batch_size, num_records, self.encoder_hidden_size)
            )
            .to_dense()
        )

        fuse_embs = torch.cat([query_emb, corpus_embs], dim=-1)
        return self.fuse_linear(fuse_embs)

    def _fuse_concat_token(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        record_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_records = record_mask.shape

        query_emb = self._compute_sentence_embedding(query_inputs)
        corpus_embs = [
            self._compute_sentence_embedding(doc_inputs)
            for doc_batches in corpus_inputs
            for doc_inputs in doc_batches
        ]

        corpus_embs = (
            torch.sparse_coo_tensor(
                record_mask.nonzero().T,
                torch.cat(corpus_embs),
                size=(batch_size, num_records, self.encoder_hidden_size)
            )
            .to_dense()
        )

        fuse_embs = torch.cat([query_emb.unsqueeze(dim=1), corpus_embs], dim=1)
        fuse_mask = torch.cat(
            [torch.ones_like(record_mask[:, :1]), record_mask], dim=1
        )
        return fuse_embs, fuse_mask

    def _fuse_cross_attention(
        self,
        query_inputs: BatchEncoding,
        corpus_inputs: list[list[BatchEncoding]],
        record_mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_records = record_mask.shape

        query_mask = query_inputs["attention_mask"].bool()
        query_token_embs = self.encoder(**query_inputs).last_hidden_state
        fuse_embs = []

        for index, document_batches in enumerate(corpus_inputs):
            for document_inputs in document_batches:
                num_documents = document_inputs["attention_mask"].shape[0]
                document_mask = (
                    document_inputs["attention_mask"].unsqueeze(dim=-1)
                )
                document_token_embs = (
                    self.encoder(**document_inputs).last_hidden_state
                )

                attn_out, _ = self.fuse_attn(
                    document_token_embs,
                    query_token_embs[index].expand(num_documents, -1, -1),
                    query_token_embs[index].expand(num_documents, -1, -1),
                    key_padding_mask=(
                        ~query_mask[index].expand(num_documents, -1)
                    )
                )
                fuse_emb = (
                    (attn_out * document_mask).sum(dim=1)
                    / document_mask.sum(dim=1)
                )
                fuse_embs.append(fuse_emb)

        fuse_embs = (
            torch.sparse_coo_tensor(
                record_mask.nonzero().T,
                torch.cat(fuse_embs),
                size=(batch_size, num_records, self.encoder_hidden_size)
            )
            .to_dense()
        )
        return fuse_embs

    def _compute_sentence_embedding(
        self,
        sentence_inputs: BatchEncoding
    ) -> torch.Tensor:
        mask = sentence_inputs["attention_mask"].unsqueeze(dim=-1)
        token_embs = self.encoder(**sentence_inputs).last_hidden_state
        sentence_embs = (token_embs * mask).sum(dim=1) / mask.sum(dim=1)
        return sentence_embs
