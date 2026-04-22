"""Data loading module for SMS Spam Collection dataset."""

from .loader import load_spam_data, split_data, save_processed_data

__all__ = ["load_spam_data", "split_data", "save_processed_data"]
