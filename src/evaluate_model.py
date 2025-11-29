import joblib
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

from .config import MODEL_PATH
from .data_utils import load_raw_data, preprocess_labels, train_test_split_urls


def evaluate():
    df = load_raw_data()
    df = preprocess_labels(df)
    X_train, X_test, y_train, y_test = train_test_split_urls(df)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train it first with `python -m src.train_model`."
        )

    model = joblib.load(MODEL_PATH)

    # Some linear SVC versions don't have predict_proba, but they have decision_function
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    else:
        raise AttributeError("Model does not support decision_function or predict_proba.")

    auc = roc_auc_score(y_test, scores)
    print(f"[INFO] ROC-AUC: {auc:.4f}")

    RocCurveDisplay.from_predictions(y_test, scores)
    plt.title("PhishShield ROC Curve (URL model)")
    plt.show()


if __name__ == "__main__":
    evaluate()
