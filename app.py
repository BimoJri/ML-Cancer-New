"""Versi Streamlit Cloud dari LungCare ML."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "models" / "metadata.json").read_text())
MODEL = joblib.load(ROOT / "models" / "best_model.pkl")
ENCODER = joblib.load(ROOT / "models" / "label_encoder.pkl")

st.set_page_config(page_title="LungCare ML", page_icon="🫁", layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(145deg,#f1f8ff,#f3fffb);color:#102a43}
.stApp label,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] span,
.stApp [data-testid="stMarkdownContainer"] p{color:#102a43!important}
.stApp [data-testid="stMetricLabel"] p{color:#486581!important}
.hero{padding:2.2rem;border-radius:24px;background:linear-gradient(125deg,#082d4e,#08766e);color:white;margin-bottom:2rem}
.hero h1{font-size:3rem;margin:0}.hero p{color:#d8ebee;font-size:1.1rem}.badge{display:inline-block;padding:.4rem .8rem;border-radius:999px;background:#ffffff1c}
.result{padding:1.5rem;border-radius:18px;background:white;border:2px solid #16a594;margin-top:1.5rem}
</style>""", unsafe_allow_html=True)
st.markdown("""<div class="hero"><span class="badge">✦ Deteksi dini berbasis machine learning</span>
<h1>🫁 LungCare ML</h1><p>Prediksi edukatif tingkat risiko kanker paru-paru berdasarkan 23 indikator kesehatan dan gaya hidup.</p></div>""", unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)
for col,value,label in zip((c1,c2,c3,c4),(META['dataset']['rows'],META['dataset']['features'],len(META['models']),f"{META['best_accuracy']:.0%}"),("Data pasien","Fitur","Model","Akurasi terbaik")):
    col.metric(label,value)

st.subheader("Masukkan data pasien")
st.caption("Skala 1 menunjukkan tingkat rendah; nilai maksimum menunjukkan tingkat tertinggi pada dataset.")
values={}
cols=st.columns(3)
for i,feature in enumerate(META["features"]):
    limits=META["ranges"][feature]
    with cols[i%3]:
        if feature=="Age": values[feature]=st.number_input("Age / Usia",limits["min"],limits["max"],40)
        elif feature=="Gender": values[feature]=st.selectbox("Gender",[1,2],format_func=lambda x:"Male" if x==1 else "Female")
        else: values[feature]=st.slider(feature,limits["min"],limits["max"],limits["min"])

if st.button("✦ Prediksi Sekarang",type="primary",use_container_width=True):
    row=pd.DataFrame([values],columns=META["features"])
    encoded=int(MODEL.predict(row)[0]); probs=MODEL.predict_proba(row)[0]*100
    label=str(ENCODER.inverse_transform([encoded])[0])
    advice={"Low":"Pertahankan gaya hidup sehat, hindari rokok, dan lakukan pemeriksaan berkala.","Medium":"Kurangi faktor risiko, perbaiki pola hidup, dan konsultasikan dengan dokter.","High":"Segera lakukan pemeriksaan medis lanjutan dan konsultasi dengan dokter spesialis paru."}
    st.markdown(f'<div class="result"><h2>Hasil: Risiko {label}</h2><p>{advice[label]}</p></div>',unsafe_allow_html=True)
    st.subheader("Probabilitas kelas")
    for cls,prob in sorted(zip(ENCODER.classes_,probs),key=lambda x:x[1],reverse=True):
        st.write(f"**{cls} — {prob:.2f}%**"); st.progress(float(prob/100))
    st.warning("Hasil ini bersifat edukatif dan bukan diagnosis medis.")
