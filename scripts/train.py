#!/usr/bin/env python3
"""
Main training script for SMS Spam Classification.

Orchestrates the full ML pipeline: data loading → feature extraction →
model training → evaluation → visualization → artifact persistence.
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

# Set non-interactive backend for headless environments
# (must be before pyplot import)
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402,F401

# Add project root to path for imports
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

# === Configuration ===
DEFAULT_DATA_PATH: str = 'data/raw/spam.csv'
DEFAULT_OUTPUT_DIR: str = 'artifacts'
DEFAULT_LOG_FILE: str = 'logs/training.log'
DEFAULT_RANDOM_STATE: int = 42


# === Argument Validators ===
def prob_type(value: str) -> float:
    """Validate probability value in (0, 1)."""
    fval = float(value)
    if not 0.0 < fval < 1.0:
        raise argparse.ArgumentTypeError(
            f"{value!r} not in valid range (0.0, 1.0) for test size"
        )
    return fval


def positive_int(value: str) -> int:
    """Validate positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be a positive integer"
        )
    return ivalue


# === Logging Setup ===
def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """
    Configure root logging to file and console (idempotent).

    Args:
        log_file: Path to the log file.
        level: Logging level (default: INFO).
    """
    root = logging.getLogger()

    # Prevent duplicate handlers if called multiple times (e.g., in tests)
    if root.handlers:
        root.setLevel(level)
        return

    root.setLevel(level)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # File handler with UTF-8 support
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s — %(levelname)s — %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Console handler with simpler format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    root.addHandler(fh)
    root.addHandler(ch)


# === Argument Parsing ===
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        argv: Command-line arguments (for testing; defaults to sys.argv).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Train SMS Spam Classification model',
        epilog='''
Examples:
  %(prog)s --model logistic_regression --vectorizer tfidf
  %(prog)s --test-size 0.2 --max-features 5000 --output-dir results/exp1
  %(prog)s --no-plots --log-file logs/run.log
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--data-path', type=str, default=DEFAULT_DATA_PATH,
        help='Path to the raw CSV dataset'
    )
    parser.add_argument(
        '--test-size', type=prob_type, default=0.33,
        help='Proportion of data for testing (0.0–1.0)'
    )

    # Feature extraction arguments
    parser.add_argument(
        '--vectorizer', type=str, default='count',
        choices=['count', 'tfidf'],
        help='Text vectorization method'
    )
    parser.add_argument(
        '--max-features', type=positive_int, default=2000,
        help='Maximum number of features for vectorizer'
    )

    # Model arguments
    parser.add_argument(
        '--model', type=str, default='naive_bayes',
        choices=['naive_bayes', 'logistic_regression'],
        help='Classifier type to train'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help='Directory to save artifacts (model, vectorizer, metrics, plots)'
    )
    parser.add_argument(
        '--log-file', type=str, default=DEFAULT_LOG_FILE,
        help='Path to training log file'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating visualization plots'
    )
    parser.add_argument(
        '--random-state', type=int, default=DEFAULT_RANDOM_STATE,
        help='Random seed for reproducibility'
    )

    return parser.parse_args(argv)


# === Main Pipeline ===
def main(argv: Optional[list[str]] = None) -> int:
    """
    Execute the full training pipeline.

    Args:
        argv: Command-line arguments (for testing; defaults to sys.argv).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    args = parse_args(argv)

    # Configure logging (idempotent)
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("Starting SMS Spam Classification training pipeline")
    logger.debug("Arguments: %s", vars(args))

    try:
        # === 1. Load and prepare data ===
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

        # === 2. Feature extraction ===
        logger.info("Step 2/6: Vectorizing text...")
        # ← Убран random_state: векторизаторы sklearn его не принимают
        vectorizer, X_train = fit_vectorizer(
            X_train_text.tolist(),
            method=args.vectorizer,
            max_features=args.max_features,
        )
        X_test = vectorizer.transform(X_test_text.tolist())
        logger.info("Vectorized: %d features", X_train.shape[1])

        # === 3. Model training ===
        logger.info("Step 3/6: Training %s model...", args.model)
        model = train_model(
            X_train, y_train,
            model_type=args.model,
            random_state=args.random_state
        )
        train_acc = model.score(X_train, y_train)
        logger.info("Train accuracy: %.4f", train_acc)

        # === 4. Evaluation ===
        logger.info("Step 4/6: Evaluating model...")
        y_pred, y_proba = predict_with_proba(model, X_test)
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        log_metrics(metrics, prefix='Test')

        # === 5. Save artifacts ===
        logger.info("Step 5/6: Saving artifacts...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save model and vectorizer
        model_path = output_dir / f'model_{args.model}_{timestamp}.joblib'
        vectorizer_path = (
            output_dir / f'vectorizer_{args.vectorizer}_{timestamp}.joblib'
        )
        save_model(model, model_path, overwrite=True)
        save_vectorizer(vectorizer, vectorizer_path, overwrite=True)

        # Save metrics as JSON
        metrics_path = output_dir / f'metrics_{timestamp}.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        # ← Опционально: сохранение семпла предсказаний (можно вынести за флаг)
        sample_df = pd.DataFrame({
            'text': X_test_text.iloc[:100].tolist(),
            'label': y_test.iloc[:100].tolist(),
            'predicted': y_pred[:100],
            'probability': y_proba[:100]
        })
        sample_path = output_dir / f'sample_predictions_{timestamp}.csv'
        sample_df.to_csv(sample_path, index=False)

        logger.info("Artifacts saved to %s/", output_dir)

        # === 6. Visualization (optional) ===
        if not args.no_plots:
            logger.info("Step 6/6: Generating visualizations...")
            plots_dir = output_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)

            # Confusion matrix
            cm_path = plots_dir / f'confusion_matrix_{timestamp}.png'
            plot_cm(y_test, y_pred, output_path=cm_path, show_plot=False)

            # Word clouds
            spam_texts = df[df['target'] == 1]['text']
            ham_texts = df[df['target'] == 0]['text']
            plot_wordcloud(
                spam_texts,
                title='Spam Words',
                output_path=plots_dir / f'wordcloud_spam_{timestamp}.png',
                show_plot=False
            )
            plot_wordcloud(
                ham_texts,
                title='Ham Words',
                output_path=plots_dir / f'wordcloud_ham_{timestamp}.png',
                show_plot=False
            )

            # Misclassification analysis
            errors = analyze_misclassifications(
                X_test_text, y_test, y_pred, y_proba, max_samples=10
            )
            print_misclassified_samples(errors)

            logger.info("Plots saved to %s/", plots_dir)

        # === Summary ===
        logger.info("✅ Training pipeline completed successfully!")
        logger.info(
            "Test metrics: F1=%.3f, AUC=%s",
            metrics['f1_score'],
            metrics.get('roc_auc', 'N/A')
        )
        logger.info("Artifacts: %s/", output_dir)

        return 0

    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        return 2
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 3
    except Exception:
        logger.exception("Unexpected error in pipeline:")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
