"""
Data loading and preprocessing module for SMS Spam Collection dataset.

This module provides functions to load, clean, and split the SMS Spam
Collection dataset from UCI/Kaggle.
"""

import logging
from pathlib import Path
from typing import Tuple, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

# === Configuration constants ===
DEFAULT_ENCODING: str = 'ISO-8859-1'
DEFAULT_TEST_SIZE: float = 0.33
DEFAULT_RANDOM_STATE: int = 42

# Column names for the processed DataFrame
LABEL_COLUMN: str = 'label'      # 'ham' or 'spam'
TEXT_COLUMN: str = 'text'        # SMS message content
TARGET_COLUMN: str = 'target'    # Binary: 0 = ham, 1 = spam

# Mapping from original labels to binary targets
LABEL_MAPPING: dict[str, int] = {'ham': 0, 'spam': 1}

# Supported file formats for saving processed data
SUPPORTED_FORMATS: tuple[str, ...] = ('csv', 'parquet')

logger = logging.getLogger(__name__)


def load_spam_data(
    filepath: str | Path,
    encoding: str = DEFAULT_ENCODING,
) -> pd.DataFrame:
    """
    Load and preprocess the SMS Spam Collection dataset.

    Reads the original CSV file (with columns v1=label, v2=text),
    cleans unnecessary columns, and creates a binary target variable.

    Args:
        filepath: Path to the CSV file.
        encoding: File encoding (default: 'ISO-8859-1').

    Returns:
        pd.DataFrame with columns: [text, label, target].

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the data contains unexpected label values.
        pd.errors.ParserError: If the CSV file cannot be parsed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info(f"Loading data from {filepath}")

    # Read CSV with error handling
    try:
        # Original dataset has columns:
        # v1 (label), v2 (text), + possible Unnamed cols
        df = pd.read_csv(filepath, encoding=encoding, usecols=[0, 1])
        df.columns = [LABEL_COLUMN, TEXT_COLUMN]
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(
            f"Failed to parse CSV file '{filepath}': {e}"
        ) from e
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"Encoding error reading '{filepath}'. Try a different encoding."
        )

    # Create binary target variable
    df[TARGET_COLUMN] = df[LABEL_COLUMN].map(LABEL_MAPPING)

    # Validate that all labels were mapped correctly
    if df[TARGET_COLUMN].isna().any():
        invalid_labels = df.loc[
            df[TARGET_COLUMN].isna(), LABEL_COLUMN
        ].unique().tolist()
        raise ValueError(
            f"Unexpected label values: {invalid_labels}. "
            f"Expected: {list(LABEL_MAPPING.keys())}"
        )

    # Ensure target is integer type for ML compatibility
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    logger.info(
        f"Loaded {len(df)} samples. "
        f"Spam ratio: {df[TARGET_COLUMN].mean():.2%}"
    )
    return df


def split_data(
    df: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
    target_column: str = TARGET_COLUMN,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.

    Args:
        df: DataFrame containing the data.
        text_column: Name of the column with message text.
        target_column: Name of the column with target variable.
        test_size: Proportion for test split (0.0–1.0).
        random_state: Random seed for reproducibility.
        stratify: Whether to preserve class distribution.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as pd.Series.

    Raises:
        ValueError: If required columns are missing or test_size is invalid.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame")

    missing_cols = [
        col for col in [text_column, target_column]
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            f"test_size must be between 0.0 and 1.0, got {test_size}"
        )

    stratify_param = df[target_column] if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column],
        df[target_column],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )

    logger.info(f"Split complete — Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def save_processed_data(
    df: pd.DataFrame,
    output_path: str | Path,
    file_format: Literal['csv', 'parquet'] = 'csv',
) -> None:
    """
    Save a processed DataFrame to disk.

    Args:
        df: DataFrame to save.
        output_path: Destination file path.
        file_format: Output format ('csv' or 'parquet').

    Raises:
        ValueError: If file_format is unsupported.
        ImportError: If parquet dependencies are missing.
        OSError: If the file cannot be written.
    """
    if file_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: '{file_format}'. "
            f"Supported: {SUPPORTED_FORMATS}"
        )

    if df.empty:
        logger.warning("Saving empty DataFrame to %s", output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if file_format == 'csv':
            df.to_csv(output_path, index=False)
        elif file_format == 'parquet':
            try:
                df.to_parquet(output_path, index=False)
            except ImportError as e:
                raise ImportError(
                    "Saving to parquet requires 'pyarrow' or 'fastparquet'. "
                    "Install with: pip install pyarrow"
                ) from e

        logger.info(f"Saved data to {output_path}")

    except OSError as e:
        raise OSError(f"Failed to write file '{output_path}': {e}") from e
