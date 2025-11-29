from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from .config import MODEL_PATH
from .data_utils import load_raw_data, preprocess_labels, train_test_split_urls


def build_pipeline():
    """
    Build a scikit-learn Pipeline:
        URL (string) --> TF-IDF (char n-grams) --> LinearSVC
    """
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),   # 3- to 5-character n-grams
        min_df=5,             # ignore extremely rare n-grams
        max_features=200000   # cap feature space
    )

    classifier = LinearSVC()

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    return pipeline


def train():
    # Load & preprocess data
    df = load_raw_data()
    df = preprocess_labels(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split_urls(df)

    # Build pipeline
    pipeline = build_pipeline()

    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)
    print("[INFO] Training complete.")

    print("[INFO] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)

    print("\n[REPORT] Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("[REPORT] Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
