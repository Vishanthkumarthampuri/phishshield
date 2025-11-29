from flask import Flask, render_template, request
import joblib
from pathlib import Path

from .config import MODEL_PATH


app = Flask(__name__)

# Load the trained model
MODEL_PATH_RESOLVED = Path(MODEL_PATH)
model = joblib.load(MODEL_PATH_RESOLVED)


def predict_url(url: str):
    pred = model.predict([url])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba([url])[0]
        proba = float(proba_all[int(pred)])

    label = "Phishing (bad)" if pred == 1 else "Legitimate (good)"
    return int(pred), label, proba


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_label = None
    prediction_raw = None
    probability = None
    url_input = ""

    if request.method == "POST":
        url_input = request.form.get("url", "").strip()
        if url_input:
            pred_raw, pred_label, proba = predict_url(url_input)
            prediction_label = pred_label
            prediction_raw = pred_raw
            if proba is not None:
                probability = round(proba * 100, 2)

    return render_template(
        "index.html",
        url_input=url_input,
        prediction_label=prediction_label,
        prediction_raw=prediction_raw,
        probability=probability,
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)