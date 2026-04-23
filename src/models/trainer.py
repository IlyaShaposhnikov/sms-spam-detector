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
                'alpha': NB_DEFAULT_ALPHA,
                'fit_prior': NB_DEFAULT_FIT_PRIOR,
                **nb_kwargs  # ← filtered kwargs
            }
            return MultinomialNB(**config)

        case 'logistic_regression':
            # LogisticRegression uses random_state for reproducibility
            config = {
                'max_iter': LR_DEFAULT_MAX_ITER,
                'random_state': DEFAULT_RANDOM_STATE,
                'class_weight': LR_DEFAULT_CLASS_WEIGHT,
                'n_jobs': LR_DEFAULT_N_JOBS,
                **kwargs  # ← all kwargs passed through
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
    ...
    """
    # Domain-specific validation: ensure binary labels for SMS classification
    y_array = np.asarray(y_train)
    unique_labels = np.unique(y_array)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels [0, 1], got: {unique_labels}"
        )

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
        to P(spam). This assumption holds for sklearn binary classifiers.

    Args:
        model: Fitted classifier instance.
        X: Feature matrix for prediction.

    Returns:
        tuple: (predicted_labels, probability_of_spam)

    Raises:
        ValueError: If model is not fitted or lacks predict_proba method.
    """
    check_is_fitted(model)

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
    ...
    """
    filepath = Path(filepath)

    if filepath.suffix not in SUPPORTED_MODEL_EXTENSIONS:
        filepath = filepath.with_suffix('.joblib')

    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. Use overwrite=True to force."
        )

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
    ...
    Warning:
        Only load models from trusted sources. joblib.load() can execute
        arbitrary code. For untrusted files, validate file integrity first.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath)

    # Ensure the loaded model is actually fitted
    check_is_fitted(model)

    logger.info("Loaded model from %s", filepath)
    return model
