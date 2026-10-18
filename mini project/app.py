from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "patient_risk_dataset.csv"
FEATURE_COLUMNS = [
    "age",
    "bmi",
    "heart_rate",
    "sys_bp",
    "oxygen_saturation",
]
TARGETS = {"readmission_risk": "readmission", "icu_emergency_risk": "icu"}
MODEL_STATE = {
    "readmission": None,
    "icu": None,
}
MODEL_SUMMARY = {
    "readmission": {
        "trained": False,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
    },
    "icu": {
        "trained": False,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "confusion_matrix": [[0, 0], [0, 0]],
    },
}


def ensure_dataset_exists() -> None:
    if not DATASET_PATH.exists():
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.run(["python", str(BASE_DIR / "data" / "generate_patient_risk_data.py")], check=True)


def get_model_pipeline() -> Pipeline:
    numeric_features = FEATURE_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_classif, k=len(numeric_features))),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ]
    )


def prepare_dataset(file_path: str | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        ensure_dataset_exists()
        df = pd.read_csv(DATASET_PATH)

    missing_cols = [col for col in FEATURE_COLUMNS + list(TARGETS.keys()) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    return df, FEATURE_COLUMNS, list(TARGETS.keys())


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
    report = classification_report(y_test, predictions, labels=[0, 1], zero_division=0, output_dict=True)

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def train_models(file_path: str | None = None) -> dict[str, Any]:
    df, feature_columns, target_columns = prepare_dataset(file_path)
    models: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    for target_col, key in TARGETS.items():
        X = df[feature_columns]
        y = df[target_col].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        pipeline = get_model_pipeline()
        pipeline.fit(X_train, y_train)
        models[key] = pipeline
        summary[key] = evaluate_model(pipeline, X_test, y_test)

    return {"models": models, "summary": summary}


def predict_patient_risk(patient_data: dict[str, Any]) -> dict[str, Any]:
    data = pd.DataFrame([patient_data])
    outputs = {}
    for model_key, model in MODEL_STATE.items():
        probability = model.predict_proba(data)[0]
        positive_index = 1
        chance = float(probability[positive_index])
        label = "HIGH" if chance >= 0.5 else "LOW"
        outputs[model_key] = {
            "label": label,
            "probability": round(float(chance), 4),
            "confidence_percent": round(float(chance * 100), 2),
            "risk_class": label,
            "class_distribution": {
                "LOW": round(float(probability[0]), 4),
                "HIGH": round(float(probability[1]), 4),
            },
            "display_label": f"{label} {'READMISSION' if model_key == 'readmission' else 'ICU EMERGENCY'} RISK",
        }

    return outputs


@app.route("/")
def index():
    return render_template("index.html", model_summary=MODEL_SUMMARY)


@app.route("/api/train", methods=["POST"])
def train_models_route():
    global MODEL_STATE, MODEL_SUMMARY

    uploaded_file = request.files.get("datasetFile")
    file_path = None
    if uploaded_file and uploaded_file.filename:
        upload_dir = BASE_DIR / "uploads"
        upload_dir.mkdir(exist_ok=True)
        file_path = str(upload_dir / uploaded_file.filename)
        uploaded_file.save(file_path)

    results = train_models(file_path)
    MODEL_STATE = results["models"]
    MODEL_SUMMARY = {
        "readmission": {
            "trained": True,
            "accuracy": results["summary"]["readmission"]["accuracy"],
            "precision": results["summary"]["readmission"]["precision"],
            "recall": results["summary"]["readmission"]["recall"],
            "f1_score": results["summary"]["readmission"]["f1_score"],
            "confusion_matrix": results["summary"]["readmission"]["confusion_matrix"],
        },
        "icu": {
            "trained": True,
            "accuracy": results["summary"]["icu"]["accuracy"],
            "precision": results["summary"]["icu"]["precision"],
            "recall": results["summary"]["icu"]["recall"],
            "f1_score": results["summary"]["icu"]["f1_score"],
            "confusion_matrix": results["summary"]["icu"]["confusion_matrix"],
        },
    }

    return jsonify({
        "message": "Model training completed successfully.",
        "summary": MODEL_SUMMARY,
    })


@app.route("/api/predict", methods=["POST"])
def predict_route():
    global MODEL_STATE, MODEL_SUMMARY

    if not MODEL_STATE["readmission"] or not MODEL_STATE["icu"]:
        results = train_models()
        MODEL_STATE = results["models"]
        MODEL_SUMMARY = {
            "readmission": {
                "trained": True,
                "accuracy": results["summary"]["readmission"]["accuracy"],
                "precision": results["summary"]["readmission"]["precision"],
                "recall": results["summary"]["readmission"]["recall"],
                "f1_score": results["summary"]["readmission"]["f1_score"],
                "confusion_matrix": results["summary"]["readmission"]["confusion_matrix"],
            },
            "icu": {
                "trained": True,
                "accuracy": results["summary"]["icu"]["accuracy"],
                "precision": results["summary"]["icu"]["precision"],
                "recall": results["summary"]["icu"]["recall"],
                "f1_score": results["summary"]["icu"]["f1_score"],
                "confusion_matrix": results["summary"]["icu"]["confusion_matrix"],
            },
        }

    form_data = request.form.to_dict()
    patient_record = {}
    for field in FEATURE_COLUMNS:
        value = form_data.get(field)
        if value in (None, ""):
            patient_record[field] = 0
            continue
        try:
            patient_record[field] = float(value)
        except ValueError:
            patient_record[field] = 0

    predictions = predict_patient_risk(patient_record)
    return jsonify({
        "message": "Prediction Completed",
        "prediction": predictions,
        "model_summary": MODEL_SUMMARY,
    })


if __name__ == "__main__":
    ensure_dataset_exists()
    results = train_models()
    MODEL_STATE = results["models"]
    MODEL_SUMMARY = {
        "readmission": {
            "trained": True,
            "accuracy": results["summary"]["readmission"]["accuracy"],
            "precision": results["summary"]["readmission"]["precision"],
            "recall": results["summary"]["readmission"]["recall"],
            "f1_score": results["summary"]["readmission"]["f1_score"],
            "confusion_matrix": results["summary"]["readmission"]["confusion_matrix"],
        },
        "icu": {
            "trained": True,
            "accuracy": results["summary"]["icu"]["accuracy"],
            "precision": results["summary"]["icu"]["precision"],
            "recall": results["summary"]["icu"]["recall"],
            "f1_score": results["summary"]["icu"]["f1_score"],
            "confusion_matrix": results["summary"]["icu"]["confusion_matrix"],
        },
    }
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)
