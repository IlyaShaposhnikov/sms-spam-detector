"""
Visualization module for SMS spam classification analysis.

Provides functions to generate word clouds and analyze misclassified
samples for model interpretability and debugging.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

# === Configuration constants ===
DEFAULT_DPI: int = 100
DEFAULT_WC_WIDTH: int = 800
DEFAULT_WC_HEIGHT: int = 400
DEFAULT_WC_BACKGROUND: str = "white"
DEFAULT_WC_COLORMAP: str = "viridis"
WC_DEFAULT_MAX_WORDS: int = 100

# Misclassification analysis defaults
MAX_MISCLASSIFIED_SAMPLES: int = 10
MAX_PREVIEW_LENGTH: int = 150

logger = logging.getLogger(__name__)


def plot_wordcloud(
    texts: Union[list[str], pd.Series],
    title: str = "Word Cloud",
    width: int = DEFAULT_WC_WIDTH,
    height: int = DEFAULT_WC_HEIGHT,
    background_color: str = DEFAULT_WC_BACKGROUND,
    colormap: str = DEFAULT_WC_COLORMAP,
    max_words: int = WC_DEFAULT_MAX_WORDS,
    stopwords: Optional[set[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> WordCloud:
    """
    Generate and display a word cloud from text data.

    Args:
        texts: List or Series of text documents to visualize.
        title: Plot title.
        width/height: WordCloud dimensions in pixels.
        background_color: Background color for the cloud.
        colormap: Matplotlib colormap name for word coloring.
        max_words: Maximum number of words to display.
        stopwords: Set of words to exclude. If None, uses wordcloud.STOPWORDS.
        output_path: If provided, save plot to this path instead of showing.
        show_plot: Whether to display the plot with plt.show().

    Returns:
        WordCloud instance for further customization (e.g., wc.words_).

    Raises:
        ValueError: If texts is empty or contains no valid content.
    """
    # Convert pandas Series to list for uniform processing
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    if not texts:
        raise ValueError("Cannot create wordcloud from empty text list")

    # Build corpus
    text_corpus = " ".join(str(t).lower() for t in texts if pd.notna(t))
    if not text_corpus.strip():
        raise ValueError("No valid text content found for wordcloud")

    # Generate word cloud with configured parameters
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap=colormap,
        max_words=max_words,
        min_font_size=10,
        random_state=42,
        # Filter common English words (the, a, is, etc.) that add noise
        stopwords=stopwords if stopwords is not None else STOPWORDS,
    ).generate(text_corpus)

    plt.figure(figsize=(width / 100, height / 100), dpi=DEFAULT_DPI)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved wordcloud to %s", output_path)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return wc


def analyze_misclassifications(
    texts: pd.Series,
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: Optional[NDArray] = None,
    max_samples: int = MAX_MISCLASSIFIED_SAMPLES,
    preview_length: int = MAX_PREVIEW_LENGTH,
) -> dict[str, list[dict]]:
    """
    Analyze and return misclassified samples for inspection.

    Args:
        texts: Series of text messages.
        y_true: Ground truth labels (0=ham, 1=spam).
        y_pred: Predicted labels (0=ham, 1=spam).
        y_proba: Predicted probabilities for spam class (optional).
        max_samples: Maximum number of misclassified samples
        to return per category.
        preview_length: Maximum characters for text preview in output.

    Returns:
        Dictionary with 'false_positives' and 'false_negatives' lists.

    Note:
        Input validation is delegated to numpy/sklearn for cleaner code.
    """
    # Minimal validation: probabilities must match length if provided
    if y_proba is not None and len(y_proba) != len(y_true):
        raise ValueError("y_proba length must match y_true length")

    # Convert to numpy arrays for efficient boolean indexing
    texts_arr = (
        texts.values
        if isinstance(texts, pd.Series)
        else np.asarray(texts)
    )
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Identify all misclassified samples
    misclassified_mask = y_true_arr != y_pred_arr
    if not np.any(misclassified_mask):
        logger.info("No misclassifications found — perfect predictions!")
        return {"false_positives": [], "false_negatives": []}

    # False positives: actual ham (0) but predicted as spam (1)
    # These are legitimate messages incorrectly blocked — high user impact
    fp_mask = (y_true_arr == 0) & (y_pred_arr == 1)

    # False negatives: actual spam (1) but predicted as ham (0)
    # These are spam messages that slipped through — security risk
    fn_mask = (y_true_arr == 1) & (y_pred_arr == 0)

    def _extract_samples(mask: NDArray[np.bool_]) -> list[dict]:
        indices = np.where(mask)[0][:max_samples]
        samples = []
        for idx in indices:
            text_str = str(texts_arr[idx])
            sample = {
                "index": int(idx),
                "text": text_str,
                "preview": (
                    text_str[:preview_length] + "..."
                    if len(text_str) > preview_length
                    else text_str
                ),
            }
            if y_proba is not None:
                sample["probability"] = float(y_proba[idx])
            samples.append(sample)
        return samples

    results = {
        "false_positives": _extract_samples(fp_mask),
        "false_negatives": _extract_samples(fn_mask),
    }

    logger.info(
        "Misclassification analysis: %d false positives (ham→spam), "
        "%d false negatives (spam→ham)",
        int(np.sum(fp_mask)), int(np.sum(fn_mask))
    )
    return results


def print_misclassified_samples(
    misclassifications: dict[str, list[dict]],
    include_probability: bool = True,
) -> None:
    """
    Pretty-print misclassified samples to console.

    Args:
        misclassifications: Output from analyze_misclassifications().
        include_probability: Whether to show prediction probability.

    Note:
        Console output format optimized for readability during debugging:
        - Clear section headers for FP vs FN
        - Numbered list for easy reference
        - Probability scores help prioritize which errors to investigate first
    """
    for category, key in [
        ("False Positives (ham → spam)", "false_positives"),
        ("False Negatives (spam → ham)", "false_negatives")
    ]:
        samples = misclassifications.get(key, [])
        if not samples:
            continue

        print(f"\n{category}: {len(samples)} sample(s) shown")
        print("─" * 60)
        for i, sample in enumerate(samples, 1):
            prob_part = (
                f" | prob={sample['probability']:.3f}"
                if include_probability and "probability" in sample
                else ""
            )
            print(f"{i}. {sample['preview']}{prob_part}")
