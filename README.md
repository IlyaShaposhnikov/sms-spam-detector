**[Russian Version / На русском](README.ru.md)**

# SMS Spam Detector

Machine learning project for binary classification of SMS messages as spam or ham.

## Features

- **Multiple models**: Naive Bayes (default) and Logistic Regression support
- **Flexible vectorization**: CountVectorizer or TfidfVectorizer with configurable parameters
- **CLI interface**: Easy training with command-line arguments
- **Comprehensive evaluation**: Accuracy, F1, Precision, Recall, ROC-AUC + confusion matrix
- **Interpretability**: Word clouds and misclassification analysis
- **Production-ready patterns**: Model/vectorizer persistence, structured logging, error handling
- **Modular architecture**: Clean separation of data, features, models, evaluation, visualization

## Quick Start

### 1. Clone and install
```bash
# Clone the repository
git clone https://github.com/IlyaShaposhnikov/sms-spam-detector.git
cd sms-spam-detector

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare data
Download the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and place it at:
```
data/raw/spam.csv
```

### 3. Train the model
```bash
# Default: Naive Bayes + CountVectorizer
python scripts/train.py

# With options
python scripts/train.py \
  --model logistic_regression \
  --vectorizer tfidf \
  --max-features 5000 \
  --test-size 0.25 \
  --output-dir my_experiment
```

## Advanced Usage

### Save processed data

If you want to cache preprocessed data for faster retraining or experimentation:

```python
from src.data.loader import load_spam_data, save_processed_data

# Load and preprocess
df = load_spam_data('data/raw/spam.csv')
# ... your custom preprocessing here ...

# Save in CSV or Parquet format
save_processed_data(df, 'data/processed/cleaned.csv', file_format='csv')
# or
save_processed_data(df, 'data/processed/cleaned.parquet', file_format='parquet')
```

Load it later without re-parsing the raw file:

```python
import pandas as pd

# For CSV
df = pd.read_csv('data/processed/cleaned.csv')

# For Parquet (requires pyarrow or fastparquet)
df = pd.read_parquet('data/processed/cleaned.parquet')
```

> **Why use this?**  
> - Speed up iterative development by skipping repeated I/O and cleaning  
> - Store intermediate results after custom preprocessing (e.g., lemmatization, feature engineering)  
> - Parquet format offers better compression and faster loading for large datasets

## Project Structure

```
sms-spam-detector/
├── src/
│   ├── data/             # Data loading module
│   ├── features/         # Text vectorization
│   ├── models/           # Model training & prediction
│   ├── evaluation/       # Metrics & confusion matrix
│   └── visualization/    # WordCloud & error analysis
├── scripts/
│   └── train.py          # Main training CLI
├── data/
│   ├── raw/              # Raw datasets (gitignored)
│   └── processed/        # Processed data (gitignored)
├── artifacts/            # Saved models, metrics, plots (gitignored)
├── logs/                 # Training logs (gitignored)
├── docs/                 # README images
├── .gitignore
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (English)
└── README.ru.md          # Project documentation (Russian)
```

## Configuration

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `data/raw/spam.csv` | Path to input CSV |
| `--test-size` | `0.33` | Test set proportion |
| `--vectorizer` | `count` | `count` or `tfidf` |
| `--max-features` | `2000` | Max features for vectorizer |
| `--model` | `naive_bayes` | `naive_bayes` or `logistic_regression` |
| `--output-dir` | `artifacts` | Directory for saved artifacts |
| `--log-file` | `logs/training.log` | Path to log file |
| `--no-plots` | `False` | Skip generating visualizations |
| `--random-state` | `42` | Seed for reproducibility |

### Logging

Logs are written to both console and file (`logs/training.log` by default) with format:
```
2024-04-22 14:30:22 — INFO — Step 3/6: Training naive_bayes model...
```

> **Windows users**: Paths in logs may show backslashes (`\`) instead of forward slashes (`/`). This is normal and does not affect functionality.

## Expected Results

Results may vary based on `--max-features`, `--test-size`, and random split.
With default settings (`max_features=2000`, `test_size=0.33`):

| Metric | Typical Range | Default Settings |
|--------|--------------|-------------------|
| Accuracy | 0.97–0.99 | **0.986** |
| F1 Score | 0.89–0.95 | **0.946** |
| Precision | 0.85–0.97 | **0.970** |
| Recall | 0.92–0.94 | **0.923** |
| ROC-AUC | 0.98–0.99 | **0.980** |

> Higher `--max-features` generally improves F1/Precision at the cost of training time.

### Confusion Matrix
![Confusion Matrix](docs/confusion_matrix.png)

*Test set: 1,839 samples | F1=0.946 | AUC=0.980*

### Word Clouds
| Spam | Ham |
|---------------|---------------------|
| ![Spam Words](docs/wordcloud_spam.png) | ![Ham Words](docs/wordcloud_ham.png) |

> Larger words = higher frequency in the corpus.

## Author

Ilya Shaposhnikov | [E-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

**[Russian Version / На русском](README.ru.md)**