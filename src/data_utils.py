import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    RAW_DATA_PATH,
    URL_COLUMN,
    LABEL_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Generic CSV loader (not used by training, but kept for utility).
    """
    df = pd.read_csv(file_path, encoding="latin1")
    return df


def load_raw_data() -> pd.DataFrame:
    """
    Load the phishing site URLs dataset from CSV.

    Uses a more tolerant CSV parser and skips badly formatted lines.
    Assumes the first row is the header row (column names).
    """
    print(f"[INFO] Loading data from {RAW_DATA_PATH}")

    # Robust CSV loading: use Python engine + skip bad lines
    try:
        df = pd.read_csv(
            RAW_DATA_PATH,
            encoding="latin1",
            engine="python",      # <-- important: avoid C engine ParserError
            on_bad_lines="skip",  # <-- skip rows with inconsistent columns
        )
    except TypeError:
        # For very old pandas versions that don't support on_bad_lines
        df = pd.read_csv(
            RAW_DATA_PATH,
            encoding="latin1",
            engine="python",
            error_bad_lines=False,
            warn_bad_lines=True,
        )

    print("[INFO] First 5 rows of the dataset:")
    print(df.head())

    print("[INFO] Columns in the dataset:")
    print(list(df.columns))

    # Basic sanity checks
    if URL_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected columns '{URL_COLUMN}' and '{LABEL_COLUMN}' not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please open config.py and set URL_COLUMN and LABEL_COLUMN correctly."
        )

    # Drop rows where URL or label is missing
    df = df.dropna(subset=[URL_COLUMN, LABEL_COLUMN])

    print("[INFO] Dataset shape after dropping missing values:", df.shape)
    print("[INFO] Unique label values:", df[LABEL_COLUMN].unique())

    return df


def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert label column to binary form:
        0 = legitimate
        1 = phishing

    This function tries several common mappings based on observed label values.
    If none match, you will need to manually adjust the mapping.
    """
    labels = df[LABEL_COLUMN].unique()
    print(f"[INFO] Original label values: {labels}")

    # Common mappings for popular phishing datasets.
    mapping_candidates = [
        {"good": 0, "bad": 1},
        {"benign": 0, "malicious": 1},
        {"legitimate": 0, "phishing": 1},
        {0: 0, 1: 1},
        {-1: 0, 1: 1},  # UCI style: -1 legit, 1 phishing
    ]

    for mapping in mapping_candidates:
        if set(labels).issubset(mapping.keys()):
            print(f"[INFO] Using label mapping: {mapping}")
            df[LABEL_COLUMN] = df[LABEL_COLUMN].map(mapping)
            print("[INFO] New label distribution:")
            print(df[LABEL_COLUMN].value_counts())
            return df

    # If we reach here, labels are something else (e.g. 'phishing', 'legit', etc.)
    raise ValueError(
        "Unknown label values. Please open data_utils.py -> preprocess_labels() "
        "and add a mapping for your specific label values.\n"
        f"Current label values: {labels}"
    )


def train_test_split_urls(df: pd.DataFrame):
    """
    Split the dataframe into train/test sets using the URL string as X and label as y.
    """
    # Ensure URL is string
    X = df[URL_COLUMN].astype(str).values
    y = df[LABEL_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test
