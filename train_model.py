"""Latih dan simpan tiga model klasifikasi risiko kanker paru-paru."""
from pathlib import Path
import json
import os
import joblib
ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = ROOT / "dataset" / "cancer patient data sets.csv"
MODELS = ROOT / "models"
IMAGES = ROOT / "static" / "images"
DROP_COLUMNS = ["index", "Patient Id"]
MODEL_SPECS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    ),
}

def safe_name(name):
    return name.lower().replace(" ", "_")

def save_confusion_matrix(cm, labels, name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.xlabel("Prediksi"); plt.ylabel("Aktual"); plt.tight_layout()
    plt.savefig(IMAGES / f"cm_{safe_name(name)}.png", dpi=150)
    plt.close()

def save_feature_importance(model, features, name):
    values = pd.Series(model.feature_importances_, index=features).sort_values().tail(12)
    plt.figure(figsize=(9, 6)); values.plot(kind="barh", color="#16a085")
    plt.title(f"Feature Importance — {name}", fontweight="bold")
    plt.xlabel("Importance"); plt.tight_layout()
    plt.savefig(IMAGES / f"importance_{safe_name(name)}.png", dpi=150)
    plt.close()

def main():
    MODELS.mkdir(exist_ok=True); IMAGES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    processed = data.drop(columns=DROP_COLUMNS)
    X, y = processed.drop(columns="Level"), processed["Level"]
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    metadata = {
        "dataset": {"rows": len(data), "features": X.shape[1], "columns": list(data.columns),
                    "missing": int(data.isna().sum().sum()), "duplicates": int(data.duplicated().sum()),
                    "class_distribution": {k: int(v) for k, v in y.value_counts().items()}},
        "features": list(X.columns), "classes": encoder.classes_.tolist(), "models": {},
        "ranges": {c: {"min": int(X[c].min()), "max": int(X[c].max())} for c in X.columns},
        "split": {"train": len(X_train), "test": len(X_test), "test_size": 0.2, "random_state": 42},
    }

    sns.set_theme(style="whitegrid")
    order = ["Low", "Medium", "High"]
    counts = y.value_counts().reindex(order)
    plt.figure(figsize=(8, 5)); sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                                            palette=["#22c55e", "#f59e0b", "#ef4444"], legend=False)
    plt.title("Distribusi Tingkat Risiko"); plt.xlabel("Level"); plt.ylabel("Jumlah Pasien"); plt.tight_layout()
    plt.savefig(IMAGES / "class_distribution.png", dpi=150); plt.close()

    comparison = []
    for name, model in MODEL_SPECS.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
        report = classification_report(y_test, pred, target_names=encoder.classes_, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, pred).tolist()
        metadata["models"][name] = {**{k: round(float(v), 6) for k, v in metrics.items()},
                                    "report": report, "confusion_matrix": cm,
                                    "file": f"{safe_name(name)}.pkl"}
        comparison.append({"model": name, **metrics})
        joblib.dump(model, MODELS / f"{safe_name(name)}.pkl")
        save_confusion_matrix(np.array(cm), encoder.classes_, name)
        raw_model = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(raw_model, "feature_importances_"):
            save_feature_importance(raw_model, X.columns, name)

    comparison.sort(key=lambda item: (item["accuracy"], item["f1"]), reverse=True)
    metadata["best_model"] = comparison[0]["model"]
    metadata["best_accuracy"] = round(float(comparison[0]["accuracy"]), 6)
    metadata["comparison"] = [{k: round(float(v), 6) if k != "model" else v for k, v in row.items()}
                              for row in comparison]
    best_file = metadata["models"][metadata["best_model"]]["file"]
    joblib.dump(joblib.load(MODELS / best_file), MODELS / "best_model.pkl")
    joblib.dump(encoder, MODELS / "label_encoder.pkl")

    plot_df = pd.DataFrame(comparison).set_index("model")
    plot_df.plot(kind="bar", figsize=(10, 6), color=["#2563eb", "#14b8a6", "#f59e0b", "#7c3aed"])
    plt.title("Perbandingan Performa Model"); plt.ylabel("Skor"); plt.ylim(0, 1.05)
    plt.xticks(rotation=0); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(IMAGES / "model_comparison.png", dpi=150); plt.close()

    (MODELS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selesai. Model terbaik: {metadata['best_model']} ({metadata['best_accuracy']:.2%})")

if __name__ == "__main__":
    main()
