"""
Feature extraction module for text vectorization.

Provides factory functions to create text vectorizers (Count/Tfidf)
and utilities to save/load fitted vectorizers for reproducibility.
"""

import logging
from pathlib import Path
from typing import Literal, Union

import joblib
# Type hint for sparse matrices (memory-efficient)
from scipy.sparse import spmatrix
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

# Default vectorization method: CountVectorizer is faster
# and often sufficient for SMS
DEFAULT_VECTORIZER_METHOD: Literal["count", "tfidf"] = "count"

# N-gram range: (1, 2) captures both unigrams ('win') and bigrams ('call now')
# Important for SMS where short phrases carry spam signals
DEFAULT_NGRAM_RANGE: tuple[int, int] = (1, 2)

# Document frequency thresholds:
# max_df=0.95: ignore terms appearing in >95% of docs (too common, low signal)
# min_df=2: ignore terms appearing in only 1 doc (likely noise/typos)
DEFAULT_MAX_DF: float = 0.95
DEFAULT_MIN_DF: int = 2

# Skip undecodable chars instead of crashing
DEFAULT_DECODE_ERROR: str = "ignore"
# Normalize accented chars (e.g., café → cafe)
DEFAULT_STRIP_ACCENTS: str = "unicode"

# TF-IDF specific: sublinear_tf applies log(1+tf) scaling
# to reduce weight of very frequent terms
# Helps prevent dominant terms from overshadowing rarer but informative words
DEFAULT_SUBLINEAR_TF: bool = True

# Supported file extensions for vectorizer persistence
# .joblib is preferred: faster and more efficient
# for numpy/scipy objects than .pkl
SUPPORTED_VECTORIZER_EXTENSIONS: tuple[str, ...] = (".pkl", ".joblib")

# Type alias for union of supported vectorizer types
VectorizerType = Union[CountVectorizer, TfidfVectorizer]

logger = logging.getLogger(__name__)


def create_vectorizer(
    method: Literal["count", "tfidf"] = DEFAULT_VECTORIZER_METHOD,
    **kwargs
) -> VectorizerType:
    """
    Create a text vectorizer instance.

    Factory function that returns either CountVectorizer or TfidfVectorizer
    with sensible defaults for SMS spam detection.

    Args:
        method: Vectorization method ('count' or 'tfidf').
        **kwargs: Additional parameters passed to the vectorizer constructor.
                  User-provided values override defaults.

    Returns:
        Configured vectorizer instance (CountVectorizer or TfidfVectorizer).

    Raises:
        ValueError: If method is not supported.

    Example:
        vec = create_vectorizer('count', max_features=1000)
        X = vec.fit_transform(['hello world', 'spam message'])
    """
    # Base defaults for SMS text classification
    defaults = {
        "decode_error": DEFAULT_DECODE_ERROR,
        "strip_accents": DEFAULT_STRIP_ACCENTS,
        "lowercase": True,
        "ngram_range": DEFAULT_NGRAM_RANGE,
        "max_df": DEFAULT_MAX_DF,
        "min_df": DEFAULT_MIN_DF,
    }

    # TF-IDF specific defaults
    if method == "tfidf":
        defaults["sublinear_tf"] = DEFAULT_SUBLINEAR_TF

    # Merge defaults with user-provided kwargs (user kwargs take precedence)
    config = {**defaults, **kwargs}

    logger.info(
        "Creating %s vectorizer with config: %s",
        method.upper(), config
    )

    match method:
        case "tfidf":
            return TfidfVectorizer(**config)
        case "count":
            return CountVectorizer(**config)
        case _:
            raise ValueError(
                f"Unsupported vectorizer method: '{method}'. "
                f"Supported: {DEFAULT_VECTORIZER_METHOD!r}, 'tfidf'"
            )


def fit_vectorizer(
    texts: list[str],
    method: Literal["count", "tfidf"] = DEFAULT_VECTORIZER_METHOD,
    **kwargs
) -> tuple[VectorizerType, spmatrix]:
    """
    Fit a vectorizer on text data and return the transformed matrix.

    Convenience function that creates, fits, and transforms in one step.

    Args:
        texts: List of text documents to vectorize.
        method: Vectorization method ('count' or 'tfidf').
        **kwargs: Additional parameters for the vectorizer.

    Returns:
        tuple: (fitted_vectorizer, transformed_sparse_matrix)

    Note:
        Input validation is delegated to sklearn vectorizers, which raise
        clear errors for empty inputs or invalid types.
    """
    vectorizer = create_vectorizer(method, **kwargs)
    X = vectorizer.fit_transform(texts)

    logger.info(
        "Vectorizer fitted: %d documents, %d features",
        X.shape[0], X.shape[1]
    )
    return vectorizer, X


def save_vectorizer(
    vectorizer: VectorizerType,
    filepath: str | Path,
    overwrite: bool = False
) -> None:
    """
    Save a fitted vectorizer to disk using joblib.

    Args:
        vectorizer: Fitted vectorizer instance to save.
        filepath: Destination path for the saved file (.pkl or .joblib).
        overwrite: Whether to overwrite existing file.

    Raises:
        FileExistsError: If file exists and overwrite=False.
        ValueError: If vectorizer is not fitted.
        OSError: If the file cannot be written.
    """
    filepath = Path(filepath)

    # Ensure proper extension
    if filepath.suffix not in SUPPORTED_VECTORIZER_EXTENSIONS:
        filepath = filepath.with_suffix(".joblib")

    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}. Use overwrite=True to force."
        )

    # Robust fitted-state check using sklearn's official utility
    try:
        check_is_fitted(vectorizer)
    except NotFittedError as e:
        raise ValueError(f"Cannot save unfitted vectorizer: {e}") from e

    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(vectorizer, filepath)
        logger.info("Saved vectorizer to %s", filepath)
    except OSError as e:
        raise OSError(f"Failed to write file '{filepath}': {e}") from e


def load_vectorizer(filepath: str | Path) -> VectorizerType:
    """
    Load a fitted vectorizer from disk.

    Args:
        filepath: Path to the saved vectorizer file.

    Returns:
        Loaded vectorizer instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        NotFittedError: If the loaded vectorizer is not in fitted state.

    Warning:
        Only load vectorizers from trusted sources. joblib.load() can execute
        arbitrary code. For untrusted files, validate file integrity first
        or use a restricted unpickler.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {filepath}")

    vectorizer = joblib.load(filepath)

    # Ensure the loaded vectorizer is actually fitted
    check_is_fitted(vectorizer)

    logger.info("Loaded vectorizer from %s", filepath)
    return vectorizer
