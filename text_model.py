import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


CLASS_NAMES = ("negative", "positive")
PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(
    reviews: list[str], max_vocabulary_size: int = 20_000
) -> dict[str, int]:
    if max_vocabulary_size < 2:
        raise ValueError("Vocabulary size must be at least 2.")
    counts: Counter[str] = Counter()
    for review in reviews:
        counts.update(tokenize(review))
    vocabulary = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
    for token, _ in counts.most_common(max_vocabulary_size - len(vocabulary)):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_review(
    review: str, vocabulary: dict[str, int], max_length: int = 256
) -> list[int]:
    if max_length < 1:
        raise ValueError("Maximum review length must be positive.")
    token_ids = [
        vocabulary.get(token, vocabulary[UNKNOWN_TOKEN])
        for token in tokenize(review)[:max_length]
    ]
    return token_ids or [vocabulary[UNKNOWN_TOKEN]]


class SentimentTextCNN(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int = 128,
        filters_per_kernel: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        self.kernel_sizes = (3, 4, 5)
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, filters_per_kernel, kernel_size)
                for kernel_size in self.kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            filters_per_kernel * len(self.kernel_sizes), len(CLASS_NAMES)
        )

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids).transpose(1, 2)
        if embedded.size(2) < max(self.kernel_sizes):
            embedded = F.pad(embedded, (0, max(self.kernel_sizes) - embedded.size(2)))
        pooled_features: list[torch.Tensor] = []
        for kernel_size, convolution in zip(self.kernel_sizes, self.convolutions):
            activations = torch.relu(convolution(embedded))
            valid_positions = (lengths - kernel_size + 1).clamp(min=1)
            positions = torch.arange(
                activations.size(2), device=token_ids.device
            ).unsqueeze(0)
            activations = activations.masked_fill(
                (positions >= valid_positions.unsqueeze(1)).unsqueeze(1),
                float("-inf"),
            )
            pooled_features.append(activations.max(dim=2).values)
        return self.classifier(self.dropout(torch.cat(pooled_features, dim=1)))


def load_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[SentimentTextCNN, dict[str, int], int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    vocabulary: dict[str, int] = checkpoint["vocabulary"]
    config = checkpoint["model_config"]
    model = SentimentTextCNN(
        vocabulary_size=len(vocabulary),
        embedding_dim=config["embedding_dim"],
        filters_per_kernel=config["filters_per_kernel"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocabulary, checkpoint["max_length"]


def predict_sentiment(
    model: SentimentTextCNN,
    review: str,
    vocabulary: dict[str, int],
    max_length: int,
    device: torch.device,
) -> tuple[str, float]:
    token_ids = torch.tensor(
        [encode_review(review, vocabulary, max_length)], dtype=torch.long, device=device
    )
    lengths = torch.tensor([token_ids.size(1)], dtype=torch.long, device=device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(token_ids, lengths), dim=1)[0]
        confidence, class_index = probabilities.max(dim=0)
    return CLASS_NAMES[class_index.item()], confidence.item()
