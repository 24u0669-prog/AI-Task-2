import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from train import load_review_splits
from transformer_model import sentiment_probabilities
from text_model import (
    PAD_TOKEN,
    UNKNOWN_TOKEN,
    SentimentTextCNN,
    build_vocabulary,
    encode_review,
    tokenize,
)


class TextModelTests(unittest.TestCase):
    def test_tokenize_normalizes_case_and_punctuation(self) -> None:
        self.assertEqual(tokenize("Loved it! Isn't bad."), ["loved", "it", "isn't", "bad"])

    def test_vocabulary_reserves_padding_and_unknown_tokens(self) -> None:
        vocabulary = build_vocabulary(["great film great"])
        self.assertEqual(vocabulary[PAD_TOKEN], 0)
        self.assertEqual(vocabulary[UNKNOWN_TOKEN], 1)
        self.assertEqual(vocabulary["great"], 2)

    def test_encode_uses_unknown_for_unseen_words_and_empty_input(self) -> None:
        vocabulary = build_vocabulary(["known word"])
        self.assertEqual(encode_review("new words", vocabulary), [1, 1])
        self.assertEqual(encode_review("!!!", vocabulary), [1])

    def test_model_returns_two_class_logits(self) -> None:
        model = SentimentTextCNN(vocabulary_size=12, embedding_dim=8, filters_per_kernel=4)
        logits = model(
            torch.tensor([[2, 0], [5, 6]]),
            torch.tensor([1, 2]),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_review_splits_are_balanced_and_repeatable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "neg").mkdir()
            (root / "pos").mkdir()
            for index in range(40):
                (root / "neg" / f"negative-{index}.txt").write_text(
                    f"negative review {index}", encoding="utf-8"
                )
                (root / "pos" / f"positive-{index}.txt").write_text(
                    f"positive review {index}", encoding="utf-8"
                )

            first = load_review_splits(root, 7, 0.2, 10, 8)
            second = load_review_splits(root, 7, 0.2, 10, 8)

        self.assertEqual(first, second)
        training, validation, testing = first
        self.assertEqual(len(training), 10)
        self.assertEqual(len(validation), 14)
        self.assertEqual(len(testing), 8)
        for split in first:
            self.assertEqual([label for _, label in split].count(0), len(split) // 2)
            self.assertEqual([label for _, label in split].count(1), len(split) // 2)

    def test_pretrained_model_scores_map_to_sentiment_labels(self) -> None:
        scores = sentiment_probabilities(
            torch.tensor([[2.0, 1.0]]),
            {0: "NEGATIVE", 1: "POSITIVE"},
        )
        self.assertGreater(scores["negative"], scores["positive"])
        self.assertAlmostEqual(sum(scores.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
