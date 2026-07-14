# LungCare ML

Website Flask untuk klasifikasi risiko kanker paru-paru menjadi **Low**, **Medium**, atau **High** dengan Logistic Regression, Decision Tree, dan Random Forest.

## Menjalankan aplikasi

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Buka `http://127.0.0.1:5000`. Jalankan kembali `train_model.py` bila dataset berubah.

Model terlatih dan metadata sudah disertakan, sehingga setelah dependensi terpasang aplikasi juga dapat langsung dijalankan dengan `python app.py`.

> Sistem ini merupakan alat edukasi dan skrining berbasis data, bukan diagnosis medis.
