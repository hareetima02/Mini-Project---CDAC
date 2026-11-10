import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load the saved artifacts once, at startup
model      = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
discretizer = joblib.load(os.path.join(BASE_DIR, "discretizer.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "feature_cols.pkl"))
bool_cols    = joblib.load(os.path.join(BASE_DIR, "bool_cols.pkl"))
cont_cols    = joblib.load(os.path.join(BASE_DIR, "cont_cols.pkl"))
classes      = joblib.load(os.path.join(BASE_DIR, "classes.pkl"))


def encode_input(raw):
    """raw: dict field_name -> raw string from the HTML form."""
    row = pd.DataFrame([raw])[feature_cols]

    for c in bool_cols:
        row[c] = row[c].map({"t": 1, "f": 0}).fillna(2).astype(int)

    row["sex"] = row["sex"].map({"M": 1, "F": 0}).fillna(2).astype(int)

    for c in cont_cols:
        row[c] = pd.to_numeric(row[c].replace("?", np.nan), errors="coerce")
        row[c] = row[c].fillna(0)

    row[cont_cols] = discretizer.transform(row[cont_cols]).astype(int)
    row = row.astype(int)
    return row


@app.route("/")
def eda():
    report = ""
    report_path = os.path.join(BASE_DIR, "static", "plots", "classification_report.txt")
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            report = f.read()
    return render_template("eda.html", report=report)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        raw = {c: request.form.get(c, "") for c in feature_cols}
        try:
            X = encode_input(raw)
            proba = model.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            prediction = model.classes_[idx]
            confidence = round(float(proba[idx]) * 100, 2)
            all_proba = sorted(
                [(cls, round(float(p) * 100, 2))
                 for cls, p in zip(model.classes_, proba)],
                key=lambda t: t[1],
                reverse=True,
            )
        except Exception as e:
            return render_template("predict.html",
                                   error=str(e),
                                   feature_cols=feature_cols)
        return render_template("predict.html",
                               prediction=prediction,
                               confidence=confidence,
                               all_proba=all_proba,
                               feature_cols=feature_cols)
    return render_template("predict.html", feature_cols=feature_cols)


if __name__ == "__main__":
    app.run(debug=True)