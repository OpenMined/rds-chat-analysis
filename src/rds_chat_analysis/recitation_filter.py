"""
Recitation detection and scoring for text generation models.

This module provides tools to measure how much of a model's output overlaps
with reference texts using high-order n-gram matching, similar to BLEU scoring.
This is useful for detecting when a model is reciting or copying from its
training data or provided context.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class RecitationScore:
    """
    Data class containing recitation scoring results.

    Attributes:
        precision: Overall precision score (total overlaps / total candidate n-grams).
        overlaps: Total number of overlapping n-grams across all n-gram sizes.
        precision_per_n: Precision score for each n-gram size.
        overlaps_per_n: Number of overlapping n-grams for each n-gram size.
        candidate_ngrams_per_n: Total number of candidate n-grams for each n-gram size.
    """

    precision: float
    overlaps: int
    precision_per_n: dict[int, float]
    overlaps_per_n: dict[int, int]
    candidate_ngrams_per_n: dict[int, int]

    def __repr__(self) -> str:
        """Return a formatted string representation with newlines for better readability."""
        return (
            f"RecitationScore(\n"
            f"    precision={self.precision:.4f},\n"
            f"    overlaps={self.overlaps},\n"
            f"    precision_per_n={self.precision_per_n},\n"
            f"    overlaps_per_n={self.overlaps_per_n},\n"
            f"    candidate_ngrams_per_n={self.candidate_ngrams_per_n}\n"
            f")"
        )


class RecitationScorer:
    """
    Computes BLEU-style precision scores to measure text overlap with reference texts.

    This class measures how much of a model's output overlaps with reference texts
    using high-order n-gram matching. This is useful for detecting recitation or
    copying behavior in language models.

    The scorer computes precision scores for different n-gram sizes and provides
    both per-n and aggregate statistics.

    Algorithm:
        For each n in [n_min, n_max]:
            1. Extract all n-grams from candidate text
            2. Extract all n-grams from reference texts (union across all references)
            3. Count overlapping n-grams between candidate and references
            4. Calculate precision_n = overlaps_n / total_candidate_ngrams_n

        Overall precision = sum(all_overlaps) / sum(all_candidate_ngrams)

    Example:
        >>> scorer = RecitationScorer(n_min=4, n_max=6)
        >>> candidate = "The quick brown fox jumps over the lazy dog"
        >>> references = ["The quick brown fox", "lazy dog sleeps"]
        >>> score = scorer.score(candidate, references)
        >>> print(f"Precision: {score.precision:.3f}")
    """

    def __init__(
        self,
        model_name: str = "openai-community/gpt2",
        n_min: int = 4,
        n_max: int = 8,
    ):
        """
        Initialize the recitation scorer.

        Args:
            model_name: Name of the HuggingFace tokenizer to use for text tokenization.
                Defaults to "openai-community/gpt2".
            n_min: Minimum n-gram size for overlap detection. Defaults to 4.
            n_max: Maximum n-gram size for overlap detection. Defaults to 8.

        Note:
            Higher n-gram sizes (6-10) are more sensitive to exact copying,
            while lower n-gram sizes (2-4) may capture more general similarities.
        """
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name
        )
        self.n_min = n_min
        self.n_max = n_max

    def _get_ngrams(self, tokens: list[int], n: int) -> Counter[tuple[int, ...]]:
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def _count_overlaps(
        self, cand_tokens: list[int], ref_tokens_list: list[list[int]]
    ) -> tuple[int, dict[int, int], dict[int, int]]:
        num_overlaps_per_n: dict[int, int] = {}
        num_candidate_ngrams_per_n: dict[int, int] = {}
        for n in range(self.n_min, self.n_max + 1):
            cand_ngrams = self._get_ngrams(cand_tokens, n)

            # Combine all reference n-grams using union operator
            ref_ngrams = Counter()
            for ref_tokens in ref_tokens_list:
                ref_ngrams |= self._get_ngrams(ref_tokens, n)

            # Count overlapping n-grams between candidate and references
            num_overlaps = sum((cand_ngrams & ref_ngrams).values())
            num_candidate_ngrams = sum(cand_ngrams.values())

            num_overlaps_per_n[n] = num_overlaps
            num_candidate_ngrams_per_n[n] = num_candidate_ngrams

        total_num_overlaps = int(np.sum(list(num_overlaps_per_n.values())))
        return total_num_overlaps, num_overlaps_per_n, num_candidate_ngrams_per_n

    def _calculate_precision(self, overlap: int, total: int) -> float:
        return overlap / total if total else 0.0

    def score(
        self,
        candidate_text: str,
        reference_texts: list[str],
        lowercase: bool = True,
    ) -> RecitationScore:
        """
        Calculate recitation score between candidate text and reference texts.

        This method tokenizes the input texts and computes n-gram overlap statistics
        to measure how much of the candidate text appears in the reference texts.

        Args:
            candidate_text: The text to analyze for potential recitation.
            reference_texts: List of reference texts to compare against
                (e.g., training data, RAG context documents).
            lowercase: Whether to convert texts to lowercase before comparison.
                Defaults to True for case-insensitive matching.

        Returns:
            RecitationScore containing precision metrics and detailed statistics.

        Example:
            >>> scorer = RecitationScorer()
            >>> score = scorer.score(
            ...     "The quick brown fox jumps",
            ...     ["The quick brown fox", "jumps over the lazy dog"]
            ... )
            >>> print(f"Precision: {score.precision:.3f}")
        """
        if lowercase:
            candidate_text = candidate_text.lower()
            reference_texts = [ref.lower() for ref in reference_texts]

        cand_tokens = self.tokenizer.encode(candidate_text, add_special_tokens=False)
        ref_tokens = [
            self.tokenizer.encode(ref, add_special_tokens=False)
            for ref in reference_texts
        ]
        return self.score_tokenized(cand_tokens, ref_tokens)

    def score_tokenized(
        self, cand_tokens: list[int], ref_tokens_list: list[list[int]]
    ) -> RecitationScore:
        """
        Calculate recitation score using pre-tokenized inputs.

        This method operates directly on token IDs, which can be useful when
        you already have tokenized texts or want to avoid repeated tokenization.

        Args:
            cand_tokens: List of token IDs for the candidate text.
            ref_tokens_list: List of token ID lists for each reference text.

        Returns:
            RecitationScore containing precision metrics and detailed statistics.
        """
        num_overlaps, num_overlaps_per_n, num_candidate_ngrams_per_n = (
            self._count_overlaps(cand_tokens, ref_tokens_list)
        )

        total_num_candidate_ngrams = sum(num_candidate_ngrams_per_n.values())
        precision = self._calculate_precision(num_overlaps, total_num_candidate_ngrams)

        # Calculate per-n precision scores
        precision_per_n = {
            n: self._calculate_precision(
                num_overlaps_per_n[n], num_candidate_ngrams_per_n[n]
            )
            for n in num_overlaps_per_n.keys()
        }

        return RecitationScore(
            precision=precision,
            overlaps=num_overlaps,
            precision_per_n=precision_per_n,
            overlaps_per_n=num_overlaps_per_n,
            candidate_ngrams_per_n=num_candidate_ngrams_per_n,
        )


if __name__ == "__main__":
    # Example usage demonstrating the recitation scorer
    scorer = RecitationScorer(n_min=6, n_max=10)

    candidate = (
        "this is a test candidate text. This is a different reference text. "
        "'this is a test candidate text' is what we want to match against. "
        "This is not a match."
    )

    references = [
        "This is a reference text.",
        "Another reference text for testing.",
        "Yet another reference text.",
        "This is a different reference text. 'this is a test candidate text' "
        "is what we want to match against.",
    ]

    score = scorer.score(candidate, references)
    print(score)
