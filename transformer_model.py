from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def load_pretrained_model(model_id: str = DEFAULT_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def sentiment_probabilities(
    logits: torch.Tensor, id2label: dict[Any, str]
) -> dict[str, float]:
    """Map a binary sentiment model's output scores to positive/negative."""
    labels = {int(index): str(label).lower() for index, label in id2label.items()}
    if len(labels) != 2 or set(labels) != {0, 1}:
        raise ValueError("The selected transformer must have two sentiment labels.")

    mapped: dict[str, float] = {}
    for index, label in labels.items():
        if "positive" in label or label in {"pos", "label_1"}:
            mapped["positive"] = index
        elif "negative" in label or label in {"neg", "label_0"}:
            mapped["negative"] = index

    if set(mapped) != {"positive", "negative"}:
        raise ValueError(
            f"Unsupported sentiment labels in pretrained model: {labels}."
        )

    probabilities = torch.softmax(logits, dim=-1)[0]
    return {
        sentiment: probabilities[index].item()
        for sentiment, index in mapped.items()
    }


def predict_with_transformer(
    review: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    maximum_length = min(512, int(model.config.max_position_embeddings))
    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_length,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
    probabilities = sentiment_probabilities(logits, model.config.id2label)
    sentiment = max(probabilities, key=probabilities.__getitem__)
    return sentiment, probabilities[sentiment], probabilities


def predict_batch_with_transformer(
    reviews: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
) -> list[tuple[str, float, dict[str, float]]]:
    if not reviews:
        return []
    maximum_length = min(512, int(model.config.max_position_embeddings))
    inputs = tokenizer(
        reviews,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_length,
        padding=True,
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits

    results = []
    for row in range(len(reviews)):
        probabilities = sentiment_probabilities(
            logits[row : row + 1], model.config.id2label
        )
        sentiment = max(probabilities, key=probabilities.__getitem__)
        results.append((sentiment, probabilities[sentiment], probabilities))
    return results
