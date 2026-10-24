import argparse
import json
import random
import stat
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from text_model import (
    CLASS_NAMES,
    SentimentTextCNN,
    build_vocabulary,
    encode_review,
)


DATASET_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
    "packages/corpora/movie_reviews.zip"
)
MODEL_CONFIG = {"embedding_dim": 128, "filters_per_kernel": 64, "dropout": 0.3}


class ReviewDataset(Dataset):
    def __init__(
        self, examples: list[tuple[str, int]], vocabulary: dict[str, int], max_length: int
    ):
        self.examples = examples
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        review, label = self.examples[index]
        return encode_review(review, self.vocabulary, self.max_length), label


def collate_reviews(
    batch: list[tuple[list[int], int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(tokens) for tokens, _ in batch], dtype=torch.long)
    token_ids = torch.zeros((len(batch), int(lengths.max().item())), dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    for row, (tokens, _) in enumerate(batch):
        token_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return token_ids, lengths, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch CNN sentiment classifier on movie reviews."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-vocabulary-size", type=int, default=20_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dataset(data_dir: Path) -> Path:
    dataset_root = data_dir / "movie_reviews"
    if (dataset_root / "pos").is_dir() and (dataset_root / "neg").is_dir():
        return dataset_root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "movie_reviews.zip"
    if not archive_path.is_file():
        temporary_path = archive_path.with_suffix(".download")
        next_report = [0]

        def report_progress(block_count: int, block_size: int, total_size: int) -> None:
            downloaded = block_count * block_size
            if total_size > 0 and downloaded >= next_report[0]:
                percent = min(100, int(downloaded * 100 / total_size))
                print(f"Downloading movie reviews: {percent}%", flush=True)
                next_report[0] = downloaded + max(1, total_size // 10)

        print(f"Downloading movie-review dataset from {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, temporary_path, report_progress)
        temporary_path.replace(archive_path)

    base_path = data_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        safe_members: list[zipfile.ZipInfo] = []
        for member in archive.infolist():
            target_path = (base_path / member.filename).resolve()
            if not target_path.is_relative_to(base_path):
                raise ValueError(f"Unsafe path in review archive: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                continue
            safe_members.append(member)
        archive.extractall(path=base_path, members=safe_members)

    if not (dataset_root / "pos").is_dir() or not (dataset_root / "neg").is_dir():
        raise RuntimeError("Review archive did not contain the expected review folders.")
    return dataset_root


def load_review_splits(
    dataset_root: Path,
    seed: int,
    validation_fraction: float,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> tuple[
    list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]
]:
    training: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    testing: list[tuple[str, int]] = []
    for label, sentiment in enumerate(CLASS_NAMES):
        source = dataset_root / ("neg" if label == 0 else "pos")
        if not source.is_dir():
            raise ValueError(
                f"Could not find the {sentiment} review folder: {source}"
            )
        review_files = sorted(source.glob("*.txt"))
        if not review_files:
            raise FileNotFoundError(f"No reviews found in {source}.")
        reviews = [
            path.read_text(encoding="utf-8")
            for path in review_files
        ]
        random.Random(seed + label).shuffle(reviews)
        test_count = max(1, int(len(reviews) * 0.1))
        test_reviews = reviews[:test_count]
        train_reviews = reviews[test_count:]
        validation_count = max(1, int(len(train_reviews) * validation_fraction))
        if len(train_reviews) - validation_count < 1:
            raise ValueError("Each class needs enough reviews for train and validation.")
        validation.extend(
            (review, label) for review in train_reviews[:validation_count]
        )
        training.extend(
            (review, label) for review in train_reviews[validation_count:]
        )
        testing.extend((review, label) for review in test_reviews)

    random.Random(seed).shuffle(training)
    random.Random(seed + 1).shuffle(validation)
    random.Random(seed + 2).shuffle(testing)

    def balanced_subset(
        examples: list[tuple[str, int]], maximum: int | None
    ) -> list[tuple[str, int]]:
        if maximum is None:
            return examples
        per_class = maximum // len(CLASS_NAMES)
        remainder = maximum % len(CLASS_NAMES)
        selected: list[tuple[str, int]] = []
        for label in range(len(CLASS_NAMES)):
            class_examples = [example for example in examples if example[1] == label]
            limit = per_class + (1 if label < remainder else 0)
            selected.extend(class_examples[:limit])
        random.Random(seed + maximum).shuffle(selected)
        return selected

    training = balanced_subset(training, max_train_samples)
    testing = balanced_subset(testing, max_test_samples)
    return training, validation, testing


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0
    for token_ids, lengths, labels in loader:
        token_ids = token_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("A data loader produced no reviews.")
    return loss_sum / total, correct / total


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    model.eval()
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)
    with torch.inference_mode():
        for token_ids, lengths, labels in loader:
            predictions = model(
                token_ids.to(device), lengths.to(device)
            ).argmax(dim=1).cpu()
            for actual, predicted in zip(labels, predictions):
                confusion[actual, predicted] += 1

    true_positive = confusion.diag().float()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    total = confusion.sum().item()
    return {
        "accuracy": true_positive.sum().item() / total if total else 0.0,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "class_names": list(CLASS_NAMES),
        "confusion_matrix": confusion.tolist(),
    }


def save_training_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Training")
    axes[0].plot(epochs, history["validation_loss"], marker="o", label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="Training")
    axes[1].plot(epochs, history["validation_accuracy"], marker="o", label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("--epochs, --batch-size, and --learning-rate must be positive.")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if args.max_train_samples is not None and args.max_train_samples < 4:
        raise ValueError("--max-train-samples must be at least 4.")
    if args.max_test_samples is not None and args.max_test_samples < 2:
        raise ValueError("--max-test-samples must be at least 2.")
    if args.max_vocabulary_size < 2 or args.max_length < 5:
        raise ValueError("Vocabulary size must be >= 2 and max length must be >= 5.")

    set_seed(args.seed)
    dataset_root = ensure_dataset(args.data_dir)
    training_examples, validation_examples, test_examples = load_review_splits(
        dataset_root,
        args.seed,
        args.validation_fraction,
        args.max_train_samples,
        args.max_test_samples,
    )
    vocabulary = build_vocabulary(
        [review for review, _ in training_examples], args.max_vocabulary_size
    )
    training_loader = DataLoader(
        ReviewDataset(training_examples, vocabulary, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_reviews,
    )
    validation_loader = DataLoader(
        ReviewDataset(validation_examples, vocabulary, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reviews,
    )
    test_loader = DataLoader(
        ReviewDataset(test_examples, vocabulary, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_reviews,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device} | training reviews: {len(training_examples)} | "
        f"validation reviews: {len(validation_examples)} | "
        f"test reviews: {len(test_examples)} | vocabulary: {len(vocabulary)}"
    )
    model = SentimentTextCNN(len(vocabulary), **MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "movie_review_sentiment_cnn.pth"
    best_validation_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, training_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train loss={train_loss:.4f}, train accuracy={train_accuracy:.4f}, "
            f"validation loss={validation_loss:.4f}, "
            f"validation accuracy={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": MODEL_CONFIG,
                    "vocabulary": vocabulary,
                    "max_length": args.max_length,
                    "best_epoch": epoch,
                    "best_validation_accuracy": validation_accuracy,
                },
                checkpoint_path,
            )

    save_training_curves(history, args.output_dir / "training_curves.png")
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, test_loader, device)
    metrics["best_epoch"] = checkpoint["best_epoch"]
    metrics["best_validation_accuracy"] = checkpoint["best_validation_accuracy"]
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"Saved model: {checkpoint_path}")
    print(f"Test metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
