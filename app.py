import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Prediction System")
st.write("Masukkan data pemohon untuk memprediksi kelayakan pinjaman.")

@st.cache_resource
def load_model():
    model  = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_model()

    st.divider()
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Model", "Random Forest")
    col_m2.metric("Accuracy", "97.2%")
    col_m3.metric("F1-Score", "0.97")
    col_m4.metric("Dataset", "4.269 baris")
    st.divider()

    st.header("📝 Data Pemohon Pinjaman")

    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("💰 Pendapatan Tahunan (₹)", min_value=0, max_value=10000000, value=5000000, step=100000)
        loan_amount = st.number_input("🏷️ Jumlah Pinjaman (₹)", min_value=0, max_value=40000000, value=10000000, step=500000)
        cibil_score = st.slider("📈 CIBIL Score", min_value=300, max_value=900, value=650)
        loan_term = st.selectbox("📅 Tenor Pinjaman (tahun)", options=[2, 4, 6, 8, 10, 12, 16, 20], index=2)

    with col2:
        dependents = st.slider("👨‍👩‍👧 Jumlah Tanggungan", min_value=0, max_value=5, value=1)
        education = st.selectbox("🎓 Pendidikan", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("💼 Wiraswasta?", options=["No", "Yes"])
        residential_assets = st.number_input("🏠 Nilai Aset Rumah (₹)", min_value=0, max_value=30000000, value=2000000, step=500000)

    col3, col4 = st.columns(2)
    with col3:
        commercial_assets = st.number_input("🏬 Nilai Aset Komersial (₹)", min_value=0, max_value=20000000, value=1000000, step=500000)
        luxury_assets = st.number_input("💎 Nilai Aset Mewah (₹)", min_value=0, max_value=20000000, value=500000, step=500000)
    with col4:
        bank_assets = st.number_input("🏛️ Nilai Aset Bank (₹)", min_value=0, max_value=20000000, value=500000, step=500000)

    st.divider()

    if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):

        # Encoding sama persis seperti di Colab
        edu_enc  = 0 if education == "Graduate" else 1
        emp_enc  = 0 if self_employed == "No" else 1

        # Feature engineering sama persis seperti di Colab
        loan_to_income = loan_amount / income if income > 0 else 0
        total_assets   = residential_assets + commercial_assets + luxury_assets + bank_assets

        # Urutan fitur HARUS sama persis dengan X_train
        features = [
            'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value',
            'education', 'self_employed',
            'loan_to_income_ratio', 'total_assets'
        ]

        data_input = pd.DataFrame([[
            dependents, income, loan_amount, loan_term, cibil_score,
            residential_assets, commercial_assets, luxury_assets, bank_assets,
            edu_enc, emp_enc,
            loan_to_income, total_assets
        ]], columns=features)

        data_scaled = pd.DataFrame(scaler.transform(data_input), columns=features)
        prediction  = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0]

        st.divider()
        st.subheader("📊 Hasil Prediksi")

        if prediction == 0:
            st.success("✅ Pinjaman **DISETUJUI (Approved)**")
        else:
            st.error("❌ Pinjaman **DITOLAK (Rejected)**")

        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Probabilitas Approved", f"{probability[0]*100:.1f}%")
        col_p2.metric("Probabilitas Rejected", f"{probability[1]*100:.1f}%")

        st.divider()
        if prediction == 0:
            st.info("💡 CIBIL score sangat baik. Profil kredit memenuhi syarat." if cibil_score >= 700 else "💡 Profil pemohon memenuhi kriteria persetujuan.")
        else:
            if cibil_score < 500:
                st.warning("💡 CIBIL score terlalu rendah. Disarankan meningkatkan skor kredit.")
            elif loan_to_income > 5:
                st.warning("💡 Rasio pinjaman terhadap pendapatan terlalu tinggi.")
            else:
                st.warning("💡 Profil pemohon belum memenuhi kriteria persetujuan.")

except FileNotFoundError:
    st.error("⚠️ File model tidak ditemukan.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")

st.divider()
st.caption("🎓 Final Project AI & Big Data 2026 | Loan Approval Prediction System")
