#!/usr/bin/env python3
"""
Main training script for SMS Spam Classification.

Orchestrates the full ML pipeline: data loading → feature extraction →
model training → evaluation → visualization → artifact persistence.

Exit codes:
    0 — Success
    1 — Unexpected error (crash)
    2 — Data/file error (e.g., missing dataset)
    3 — Configuration error (e.g., invalid argument)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
import pandas as pd

# Set non-interactive backend for headless environments (CI, Docker, servers)
# Must be set BEFORE importing pyplot to avoid display errors
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401

# Add project root to sys.path to enable imports from src/
# This allows running the script directly without installing as a package
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_spam_data, split_data  # noqa: E402
from src.features.extractor import (  # noqa: E402
    fit_vectorizer, save_vectorizer
)
from src.models.trainer import (  # noqa: E402
    train_model, predict_with_proba, save_model
)
from src.evaluation.metrics import (  # noqa: E402
    calculate_metrics, plot_cm, log_metrics
)
from src.visualization.plots import (  # noqa: E402
    plot_wordcloud, analyze_misclassifications, print_misclassified_samples
)

DEFAULT_DATA_PATH: str = "data/raw/spam.csv"
DEFAULT_OUTPUT_DIR: str = "artifacts"
DEFAULT_LOG_FILE: str = "logs/training.log"
DEFAULT_RANDOM_STATE: int = 42


def prob_type(value: str) -> float:
    """
    Validate that a string argument represents
    a probability in open interval (0, 1).

    Used for --test-size argument
    to prevent degenerate splits (0% or 100% test data).
    """
    fval = float(value)
    if not 0.0 < fval < 1.0:
        raise argparse.ArgumentTypeError(
            f"{value!r} not in valid range (0.0, 1.0) for test size"
        )
    return fval


def positive_int(value: str) -> int:
    """
    Validate that a string argument represents a positive integer (> 0).
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be a positive integer"
        )
    return ivalue


def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """
    Configure root logging to file and console (idempotent).

    Args:
        log_file: Path to the log file.
        level: Logging level (default: INFO).
    """
    root = logging.getLogger()

    # Idempotency check: skip re-configuration if handlers already exist
    # This prevents duplicate log lines
    # when setup_logging is called multiple times
    if root.handlers:
        root.setLevel(level)
        return

    root.setLevel(level)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # File handler: detailed format with timestamps,
    # UTF-8 for cross-platform safety
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler: simpler format for interactive use
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.addHandler(fh)
    root.addHandler(ch)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Args:
        argv: Optional list of arguments
        (for unit testing; defaults to sys.argv).

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train SMS Spam Classification model",
        epilog="""
Examples:
  %(prog)s --model logistic_regression --vectorizer tfidf
  %(prog)s --test-size 0.2 --max-features 5000 --output-dir results/exp1
  %(prog)s --no-plots --log-file logs/run.log
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        "--data-path", type=str, default=DEFAULT_DATA_PATH,
        help="Path to the raw CSV dataset"
    )
    parser.add_argument(
        "--test-size", type=prob_type, default=0.33,
        help="Proportion of data for testing (0.0–1.0, exclusive)"
    )

    # Feature extraction arguments
    parser.add_argument(
        "--vectorizer", type=str, default="count",
        choices=["count", "tfidf"],
        help="Text vectorization method: count (bag-of-words) or tfidf"
    )
    parser.add_argument(
        "--max-features", type=positive_int, default=2000,
        help="Maximum number of features (tokens/n-grams) for vectorizer"
    )

    # Model arguments
    parser.add_argument(
        "--model", type=str, default="naive_bayes",
        choices=["naive_bayes", "logistic_regression"],
        help=(
            "Classifier type: naive_bayes (fast) "
            "or logistic_regression (accurate)"
        )
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save artifacts (model, vectorizer, metrics, plots)"
    )
    parser.add_argument(
        "--log-file", type=str, default=DEFAULT_LOG_FILE,
        help="Path to training log file"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating visualization plots (faster, for CI/headless)"
    )
    parser.add_argument(
        "--random-state", type=int, default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility of splits and stochastic models"
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """
    Execute the full training pipeline.

    Args:
        argv: Optional list of arguments (for unit testing).

    Returns:
        Exit code: 0=success, 1=crash, 2=data error, 3=config error.
    """
    args = parse_args(argv)

    # Configure logging (idempotent — safe to call multiple times)
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting SMS Spam Classification training pipeline")
    logger.debug("Arguments: %s", vars(args))

    try:
        # Step 1/6: Load and prepare data
        logger.info("Step 1/6: Loading data...")
        df = load_spam_data(args.data_path)
        X_train_text, X_test_text, y_train, y_test = split_data(
            df,
            test_size=args.test_size,
            random_state=args.random_state
        )
        logger.info(
            "Data loaded: %d total, %d train, %d test",
            len(df), len(X_train_text), len(X_test_text)
        )

        # Step 2/6: Feature extraction (text vectorization)
        logger.info("Step 2/6: Vectorizing text...")
        vectorizer, X_train = fit_vectorizer(
            X_train_text.tolist(),
            method=args.vectorizer,
            max_features=args.max_features,
        )
        # Transform test set using the same fitted vectorizer (no refitting)
        X_test = vectorizer.transform(X_test_text.tolist())
        logger.info("Vectorized: %d features", X_train.shape[1])

        # Step 3/6: Model training
        logger.info("Step 3/6: Training %s model...", args.model)
        model = train_model(
            X_train, y_train,
            model_type=args.model,
            random_state=args.random_state
        )
        train_acc = model.score(X_train, y_train)
        logger.info("Train accuracy: %.4f", train_acc)

        # Step 4/6: Evaluation on test set
        logger.info("Step 4/6: Evaluating model...")
        # Get both hard predictions and probability estimates for spam class
        y_pred, y_proba = predict_with_proba(model, X_test)
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        log_metrics(metrics, prefix="Test")

        # Step 5/6: Save artifacts for reproducibility
        logger.info("Step 5/6: Saving artifacts...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save trained model and fitted vectorizer (for inference later)
        model_path = output_dir / f"model_{args.model}_{timestamp}.joblib"
        vectorizer_path = (
            output_dir / f"vectorizer_{args.vectorizer}_{timestamp}.joblib"
        )
        save_model(model, model_path, overwrite=True)
        save_vectorizer(vectorizer, vectorizer_path, overwrite=True)

        # Save metrics as JSON for experiment tracking / comparison
        metrics_path = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Save sample predictions for manual inspection / debugging
        sample_df = pd.DataFrame({
            "text": X_test_text.iloc[:100].tolist(),
            "label": y_test.iloc[:100].tolist(),
            "predicted": y_pred[:100],
            "probability": y_proba[:100]
        })
        sample_path = output_dir / f"sample_predictions_{timestamp}.csv"
        sample_df.to_csv(sample_path, index=False)

        logger.info("Artifacts saved to %s/", output_dir)

        # Step 6/6: Visualization (optional, skip in CI with --no-plots)
        if not args.no_plots:
            logger.info("Step 6/6: Generating visualizations...")
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Confusion matrix: visualize false positives/negatives
            cm_path = plots_dir / f"confusion_matrix_{timestamp}.png"
            plot_cm(y_test, y_pred, output_path=cm_path, show_plot=False)

            # Word clouds: interpret which words drive spam/ham classification
            # Filter actual spam messages
            spam_texts = df[df["target"] == 1]["text"]
            # Filter actual ham messages
            ham_texts = df[df["target"] == 0]["text"]
            plot_wordcloud(
                spam_texts,
                title="Spam Words",
                output_path=plots_dir / f"wordcloud_spam_{timestamp}.png",
                show_plot=False
            )
            plot_wordcloud(
                ham_texts,
                title="Ham Words",
                output_path=plots_dir / f"wordcloud_ham_{timestamp}.png",
                show_plot=False
            )

            # Misclassification analysis: inspect model errors for debugging
            errors = analyze_misclassifications(
                X_test_text, y_test, y_pred, y_proba, max_samples=10
            )
            # Print to console for immediate review
            print_misclassified_samples(errors)

            logger.info("Plots saved to %s/", plots_dir)

        # Pipeline summary
        logger.info("Training pipeline completed successfully!")
        logger.info(
            "Test metrics: F1=%.3f, AUC=%s",
            metrics["f1_score"],
            # ROC-AUC only computed if probabilities provided
            metrics.get("roc_auc", "N/A")
        )
        logger.info("Artifacts: %s/", output_dir)

        # Success exit code
        return 0

    except FileNotFoundError as e:
        # Exit code 2: data/file error (distinct from config or crash)
        logger.error("Data file not found: %s", e)
        return 2
    except ValueError as e:
        # Exit code 3: configuration/argument error (user-fixable)
        logger.error("Configuration error: %s", e)
        return 3
    except Exception:
        # Exit code 1: unexpected crash (log full traceback for debugging)
        logger.exception("Unexpected error in pipeline:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
