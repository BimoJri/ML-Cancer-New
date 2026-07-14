"""Aplikasi Flask untuk prediksi tingkat risiko kanker paru-paru."""
from pathlib import Path
import json
import joblib
import pandas as pd
from flask import Flask, flash, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = "lung-risk-local-app"
DATA = pd.read_csv(ROOT / "dataset" / "cancer patient data sets.csv")
META = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
MODEL = joblib.load(ROOT / "models" / "best_model.pkl")
ENCODER = joblib.load(ROOT / "models" / "label_encoder.pkl")

RECOMMENDATIONS = {
    "Low": "Pertahankan gaya hidup sehat, hindari rokok dan asap rokok, serta lakukan pemeriksaan kesehatan secara berkala.",
    "Medium": "Kurangi faktor risiko, perbaiki pola hidup, dan konsultasikan kondisi Anda dengan dokter untuk evaluasi lebih lanjut.",
    "High": "Segera lakukan pemeriksaan medis lebih lanjut, berkonsultasi dengan dokter spesialis paru, dan pertimbangkan CT Scan sesuai anjuran tenaga medis.",
}

@app.context_processor
def shared():
    return {"meta": META}

@app.route("/")
def home(): return render_template("home.html", active="home")

@app.route("/dashboard")
def dashboard(): return render_template("dashboard.html", active="dashboard")

@app.route("/dataset")
def dataset():
    stats = DATA.drop(columns=["index"]).describe().round(2).to_html(classes="table table-hover", border=0)
    preview = DATA.head(10).to_html(classes="table table-hover", border=0, index=False)
    return render_template("dataset.html", active="dataset", preview=preview, stats=stats)

@app.route("/preprocessing")
def preprocessing(): return render_template("preprocessing.html", active="preprocessing")

@app.route("/training")
def training(): return render_template("training.html", active="training")

@app.route("/evaluation")
def evaluation(): return render_template("evaluation.html", active="evaluation")

@app.route("/visualization")
def visualization(): return render_template("visualization.html", active="visualization")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            values = {feature: int(request.form[feature]) for feature in META["features"]}
            for feature, value in values.items():
                limits = META["ranges"][feature]
                if not limits["min"] <= value <= limits["max"]:
                    raise ValueError(f"{feature} harus antara {limits['min']}–{limits['max']}.")
            row = pd.DataFrame([values], columns=META["features"])
            encoded = int(MODEL.predict(row)[0])
            probabilities = MODEL.predict_proba(row)[0]
            label = str(ENCODER.inverse_transform([encoded])[0])
            result = {"label": label, "recommendation": RECOMMENDATIONS[label],
                      "probabilities": sorted(zip(ENCODER.classes_, probabilities * 100),
                                              key=lambda item: item[1], reverse=True)}
        except (KeyError, ValueError) as exc:
            flash(f"Input tidak valid: {exc}", "danger")
    return render_template("predict.html", active="predict", result=result, values=request.form)

@app.errorhandler(404)
def not_found(error): return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
