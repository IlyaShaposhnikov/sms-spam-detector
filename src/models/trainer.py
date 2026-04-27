"""
Model training module for text classification.

Provides a unified interface for training, predicting, and persisting
scikit-learn classifiers for SMS spam detection.
"""

import logging
from pathlib import Path
from typing import Literal, Union

import joblib
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.validation import check_is_fitted

# === Type aliases ===
ModelTypeLiteral = Literal['naive_bayes', 'logistic_regression']
ModelType = Union[MultinomialNB, LogisticRegression]
FeatureMatrix = Union[spmatrix, NDArray]
TargetVector = Union[NDArray, list]

# === Configuration constants ===
DEFAULT_MODEL_TYPE: ModelTypeLiteral = 'naive_bayes'
DEFAULT_RANDOM_STATE: int = 42

# Naive Bayes defaults
NB_DEFAULT_ALPHA: float = 1.0
NB_DEFAULT_FIT_PRIOR: bool = True

# Logistic Regression defaults
LR_DEFAULT_MAX_ITER: int = 1000
LR_DEFAULT_CLASS_WEIGHT: Literal['balanced', None] = 'balanced'
LR_DEFAULT_N_JOBS: int = -1

# Supported model types
SUPPORTED_MODELS: tuple[str, ...] = ('naive_bayes', 'logistic_regression')
SUPPORTED_MODEL_EXTENSIONS: tuple[str, ...] = ('.pkl', '.joblib')

logger = logging.getLogger(__name__)


def create_model(
    model_type: ModelTypeLiteral = DEFAULT_MODEL_TYPE,
    **kwargs
) -> ModelType:
    """
    Create a classifier instance.
    ...
    """
    logger.info("Creating %s classifier", model_type.upper())

    match model_type:
        case 'naive_bayes':
            # MultinomialNB is deterministic and doesn't accept random_state
            # Filter out model-incompatible kwargs to avoid TypeError
            nb_kwargs = {
                k: v for k, v in kwargs.items() if k != 'random_state'
            }
            config = {
                # Laplace smoothing for unseen words
                'alpha': NB_DEFAULT_ALPHA,
                # Learn class priors from imbalanced data
                'fit_prior': NB_DEFAULT_FIT_PRIOR,
                # User overrides take precedence
                **nb_kwargs
            }
            return MultinomialNB(**config)

        case 'logistic_regression':
            # LogisticRegression uses random_state for reproducibility
            config = {
                # Ensure convergence in high-dimensional space
                'max_iter': LR_DEFAULT_MAX_ITER,
                # Reproducible optimization path
                'random_state': DEFAULT_RANDOM_STATE,
                # Handle imbalanced classes automatically
                'class_weight': LR_DEFAULT_CLASS_WEIGHT,
                # Parallelize across all CPU cores
                'n_jobs': LR_DEFAULT_N_JOBS,
                # User overrides take precedence
                **kwargs
            }
            return LogisticRegression(**config)

        case _:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported: {SUPPORTED_MODELS}"
            )


def train_model(
    X_train: FeatureMatrix,
    y_train: TargetVector,
    model_type: ModelTypeLiteral = DEFAULT_MODEL_TYPE,
    **kwargs
) -> ClassifierMixin:
    """
    Train a classifier on the provided data.

    Convenience function that creates, trains, and returns a fitted model.

    Args:
        X_train: Training feature matrix (sparse or dense).
        y_train: Training target vector (binary: 0 or 1).
        model_type: Type of classifier to train.
        **kwargs: Additional parameters for the classifier.

    Returns:
        Fitted classifier instance.

    Raises:
        ValueError: If data contains invalid labels
        for binary SMS classification.
    """
    # Domain-specific validation: ensure binary labels for SMS classification
    y_array = np.asarray(y_train)
    unique_labels = np.unique(y_array)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels [0, 1], got: {unique_labels}"
        )

    # Create model with method-specific defaults + user overrides, then fit
    model = create_model(model_type, **kwargs)
    model.fit(X_train, y_array)

    logger.info(
        "Model trained: %s with %d samples",
        type(model).__name__, len(y_array)
    )
    return model


def predict_with_proba(
    model: ClassifierMixin,
    X: FeatureMatrix,
) -> tuple[NDArray, NDArray]:
    """
    Generate predictions and probability estimates
    for the positive class (spam).

    Note:
        For binary classification with labels [0, 1], proba[:, 1] corresponds
        to P(spam). This assumption holds for sklearn binary classifiers where
        classes_ attribute is [0, 1] in ascending order.

    Args:
        model: Fitted classifier instance.
        X: Feature matrix for prediction.

    Returns:
        tuple: (predicted_labels, probability_of_spam)

    Raises:
        ValueError: If model is not fitted or lacks predict_proba method.
    """
    # Ensure model is fitted before prediction
    check_is_fitted(model)

    # Not all sklearn classifiers support probability estimates
    if not hasattr(model, 'predict_proba'):
        raise ValueError(
            f"Model {type(model).__name__} does not support predict_proba."
        )

    predictions = model.predict(X)
    proba = model.predict_proba(X)

    # For binary classification: column 1 = P(class=1) = P(spam)
    probabilities = proba[:, 1]

    logger.info("Generated predictions for %d samples", len(predictions))
    return predictions, probabilities


def save_model(
    model: ClassifierMixin,
    filepath: str | Path,
    overwrite: bool = False
) -> None:
    """
    Save a fitted model to disk using joblib.

    Args:
        model: Fitted classifier instance to save.
        filepath: Destination path for the saved file (.pkl or .joblib).
        overwrite: Whether to overwrite existing file.

    Raises:
        FileExistsError: If file exists and overwrite=False.
        ValueError: If model is not fitted.
        OSError: If the file cannot be written.
    """
    filepath = Path(filepath)

    # Auto-correct extension
    if filepath.suffix not in SUPPORTED_MODEL_EXTENSIONS:
        filepath = filepath.with_suffix('.joblib')

    # Prevent accidental overwrites
    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. Use overwrite=True to force."
        )

    # Only fitted models can be used for predict/predict_proba
    try:
        check_is_fitted(model)
    except NotFittedError as e:
        raise ValueError(f"Cannot save unfitted model: {e}") from e

    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, filepath)
        logger.info("Saved model to %s", filepath)
    except OSError as e:
        raise OSError(f"Failed to write file '{filepath}': {e}") from e


def load_model(filepath: str | Path) -> ClassifierMixin:
    """
    Load a fitted model from disk.

    Args:
        filepath: Path to the saved model file.

    Returns:
        Loaded classifier instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        NotFittedError: If the loaded model is not in fitted state.

    Warning:
        Only load models from trusted sources. joblib.load() uses pickle under
        the hood, which can execute arbitrary code during deserialization.
        For untrusted files, validate file integrity first
        or use a restricted unpickler.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Only load model files from trusted sources
    model = joblib.load(filepath)

    # Ensure the loaded model is actually fitted
    check_is_fitted(model)

    logger.info("Loaded model from %s", filepath)
    return model
