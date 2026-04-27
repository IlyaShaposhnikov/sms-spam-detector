"""
Evaluation metrics module for model assessment.

Provides functions to compute classification metrics and visualize
results using scikit-learn's built-in plotting utilities.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    # sklearn's native plotting API — preferred over seaborn for compatibility
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# === Type aliases ===
MetricDict = dict[str, float]

# === Configuration constants ===
DEFAULT_FIGSIZE: tuple[int, int] = (6, 5)
DEFAULT_CMAP: str = "Blues"
DEFAULT_DPI: int = 100
# Ordered to match [0, 1] target encoding
CM_DEFAULT_LABELS: tuple[str, str] = ("ham", "spam")

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: Optional[NDArray] = None,
) -> MetricDict:
    """
    Compute classification metrics for binary spam detection.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_proba: Predicted probabilities for positive class (optional).

    Returns:
        Dictionary with metric names as keys and float values:
        accuracy, f1_score, precision, recall, and optionally roc_auc.

    Note:
        Input validation is delegated to sklearn metric functions,
        which raise clear errors for invalid inputs.
    """
    # Compute core classification metrics
    metrics: MetricDict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }

    # Compute ROC-AUC if probabilities provided
    if y_proba is not None:
        # Minimal validation: probabilities must be 1D and same length
        if y_proba.ndim != 1 or len(y_proba) != len(y_true):
            raise ValueError(
                "y_proba must be 1D array with same length as y_true"
            )
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError as e:
            logger.warning("Could not compute ROC-AUC: %s", e)

    # Log results
    auc_part = (
        f", auc={metrics['roc_auc']:.3f}" if "roc_auc" in metrics else ""
    )
    logger.info(
        "Metrics: acc=%.3f, f1=%.3f, prec=%.3f, rec=%.3f%s",
        metrics["accuracy"], metrics["f1_score"],
        metrics["precision"], metrics["recall"], auc_part
    )
    return metrics


def plot_cm(
    y_true: NDArray,
    y_pred: NDArray,
    labels: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    cmap: str = DEFAULT_CMAP,
    output_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> ConfusionMatrixDisplay:
    """
    Plot confusion matrix using sklearn's ConfusionMatrixDisplay.

    This function uses scikit-learn's native plotting API,
    ensuring compatibility with current and future sklearn versions.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        labels: List of label names for display (default: ['ham', 'spam']).
        title: Plot title.
        figsize: Figure size in inches (width, height).
        cmap: Matplotlib colormap name.
        output_path: If provided, save plot to this path instead of showing.
        show_plot: Whether to display the plot with plt.show().

    Returns:
        ConfusionMatrixDisplay instance for further customization.

    Raises:
        ValueError: If labels count is not 2 for binary classification.
    """
    display_labels = list(labels) if labels else list(CM_DEFAULT_LABELS)

    # Exactly 2 labels required
    if len(display_labels) != 2:
        raise ValueError(
            "Expected 2 labels for binary classification, "
            f"got {len(display_labels)}"
        )

    # Compute confusion matrix with consistent label order [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Create display using sklearn's built-in API —
    # preferred over manual seaborn heatmap
    # because it handles formatting, annotations,
    # and future sklearn changes automatically
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels
    )
    # Show integer counts (not percentages) for direct interpretation
    disp.plot(cmap=cmap, values_format="d", colorbar=False)

    # Customize appearance
    disp.ax_.set_title(title, fontsize=14, pad=20)
    disp.ax_.set_xlabel("Predicted Label", fontsize=10)
    disp.ax_.set_ylabel("True Label", fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        disp.figure_.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        logger.info("Saved confusion matrix to %s", output_path)

    if show_plot:
        plt.show()
    else:
        plt.close(disp.figure_)

    return disp


def log_metrics(
    metrics: MetricDict,
    prefix: str = "Evaluation",
    level: int = logging.INFO,
) -> None:
    """
    Log metrics dictionary at specified logging level.

    Args:
        metrics: Dictionary of metric name -> value.
        prefix: Prefix for log messages.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    for name, value in metrics.items():
        logger.log(level, "%s — %s: %.4f", prefix, name, value)
