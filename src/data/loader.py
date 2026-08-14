"""
src/data/loader.py
------------------
Load and inspect the AfriSenti Amharic dataset.
"""

import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_afrisenti(split: str = "train") -> pd.DataFrame:
    """
    Load AfriSenti Amharic dataset from HuggingFace.

    Parameters
    ----------
    split : str
        One of 'train', 'validation', 'test'

    Returns
    -------
    pd.DataFrame
        Dataset with 'text' and 'label' columns.
    """
    from datasets import load_dataset

    print(f"Loading AfriSenti Amharic — split: {split}")
    dataset = load_dataset(
        "shmuhammad/AfriSenti-twitter-sentiment",
        "amh",
        split=split,
        trust_remote_code=True
    )
    df = dataset.to_pandas()
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}\n")
    return df


def load_all_splits() -> tuple:
    """Load train, validation, and test splits."""
    train = load_afrisenti("train")
    val = load_afrisenti("validation")
    test = load_afrisenti("test")
    return train, val, test


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Save a processed DataFrame to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_parquet(path, index=False)
    print(f"Saved: {path}")


def load_processed(filename: str) -> pd.DataFrame:
    """Load a processed file from data/processed/."""
    path = DATA_PROCESSED / filename
    df = pd.read_parquet(path)
    print(f"Loaded: {filename} — shape: {df.shape}")
    return df
