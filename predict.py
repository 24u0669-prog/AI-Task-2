import argparse
from pathlib import Path

import torch

from text_model import load_checkpoint, predict_sentiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether a movie review is positive or negative."
    )
    parser.add_argument("review", help="Review text to classify.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/movie_review_sentiment_cnn.pth"),
        help="Path to the trained checkpoint.",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model checkpoint not found: {args.model}. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocabulary, max_length = load_checkpoint(args.model, device)
    sentiment, confidence = predict_sentiment(
        model, args.review, vocabulary, max_length, device
    )
    print(f"Sentiment: {sentiment} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()
