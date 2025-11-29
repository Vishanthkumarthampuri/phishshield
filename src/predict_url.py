import argparse
import joblib

from .config import MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Run `python -m src.train_model` first."
        )
    return joblib.load(MODEL_PATH)


def predict_single_url(url: str):
    model = load_model()
    prediction = model.predict([url])[0]

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    print(f"\n[RESULT] URL: {url}")
    print(f"[RESULT] Prediction: {label} (class = {prediction})")


def main():
    parser = argparse.ArgumentParser(
        description="PhishShield - ML-based phishing URL detector"
    )
    parser.add_argument(
        "url",
        type=str,
        help="URL to classify"
    )

    args = parser.parse_args()
    predict_single_url(args.url)


if __name__ == "__main__":
    main()
