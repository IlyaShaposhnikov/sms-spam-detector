"""Feature extraction module for text vectorization."""

from .extractor import (
    create_vectorizer,
    fit_vectorizer,
    save_vectorizer,
    load_vectorizer,
    VectorizerType,
)

__all__ = [
    "create_vectorizer",
    "fit_vectorizer",
    "save_vectorizer",
    "load_vectorizer",
    "VectorizerType",
]
