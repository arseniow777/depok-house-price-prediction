import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.predict import load_artifact, predict_price, build_features

st.set_page_config(page_title="Depok House Price Predictor", page_icon="🏠", layout="wide",
                    initial_sidebar_state="collapsed")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")


@st.cache_resource
def get_artifact():
    return load_artifact(MODEL_PATH)


artifact = get_artifact()
metrics = artifact.get("metrics", {})

st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.stApp { background-color: #ffffff; color: #111111; }
.hd-card {
    background-color: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.hd-title { font-weight: 800; font-size: 2.1rem; letter-spacing: -0.02em; }
.hd-subtitle { color: #525252; font-size: 0.95rem; }
.hd-divider { border: none; border-top: 2px solid #111111; margin: 0.75rem 0 1.5rem 0; }
.hd-label { color: #525252; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.hd-price { font-size: 2.3rem; font-weight: 800; color: #eab308; }
.hd-stat-value { font-size: 1.3rem; font-weight: 700; color: #111111; }
.hd-stat-label { color: #525252; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.stButton > button {
    background-color: #eab308;
    color: #111111;
    font-weight: 700;
    border: none;
    border-radius: 6px;
}
.stButton > button:hover { background-color: #ca8a04; color: #111111; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hd-title">DEPOK HOUSE PRICE PREDICTOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hd-subtitle">Estimasi harga rumah di wilayah Depok berdasarkan luas, lokasi, dan fasilitas — '
    'ditenagai model machine learning hasil riset.</div>', unsafe_allow_html=True)
st.markdown('<hr class="hd-divider">', unsafe_allow_html=True)

tab_predict, tab_about = st.tabs(["🔮  Prediksi", "ℹ️  Tentang & Panduan"])

with tab_predict:
    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown('<div class="hd-card">', unsafe_allow_html=True)
        st.markdown("#### Masukkan Detail Rumah")
        kamar_tidur = st.number_input("Kamar Tidur", min_value=0, max_value=10, value=3, step=1)
        kamar_mandi = st.number_input("Kamar Mandi", min_value=0, max_value=10, value=2, step=1)
        garasi = st.number_input("Garasi (jumlah mobil)", min_value=0, max_value=5, value=1, step=1)
        luas_tanah = st.number_input("Luas Tanah (m²)", min_value=10, max_value=5000, value=120, step=5)
        luas_bangunan = st.number_input("Luas Bangunan (m²)", min_value=10, max_value=5000, value=100, step=5)
        kecamatan_list = sorted(artifact["kec_target_map"].keys())
        kecamatan = st.selectbox("Kecamatan", kecamatan_list)
        predict_clicked = st.button("Prediksi Harga", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="hd-card">', unsafe_allow_html=True)
        if predict_clicked:
            price, X = predict_price(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, kecamatan, artifact)
            mae = metrics.get("MAE", price * 0.2)

            st.markdown('<div class="hd-label">Estimasi Harga</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="hd-price">Rp {price:,.0f}</div>', unsafe_allow_html=True)
            st.caption(
                f"Model: {artifact['model_name']}"
                + (f" · Test R2: {metrics['R2']:.3f} · MAPE: {metrics['MAPE']*100:.1f}%" if metrics else "")
            )

            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown('<div class="hd-stat-label">Rentang Estimasi</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="hd-stat-value">Rp {max(price-mae,0):,.0f} – Rp {price+mae:,.0f}</div>',
                            unsafe_allow_html=True)
            with s2:
                st.markdown('<div class="hd-stat-label">Harga / m² Tanah</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="hd-stat-value">Rp {price/luas_tanah:,.0f}</div>', unsafe_allow_html=True)
            with s3:
                st.markdown('<div class="hd-stat-label">Harga / m² Bangunan</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="hd-stat-value">Rp {price/luas_bangunan:,.0f}</div>', unsafe_allow_html=True)

            st.write("")
            kec_avg = artifact["kec_target_map"][kecamatan]
            depok_avg = float(np.mean(list(artifact["kec_target_map"].values())))
            comp_df = pd.DataFrame({
                "Kategori": ["Prediksi Kamu", f"Rata-rata {kecamatan}", "Rata-rata Depok"],
                "Harga": [price, kec_avg, depok_avg],
            })
            fig = px.bar(comp_df, x="Kategori", y="Harga", color="Kategori",
                         color_discrete_sequence=["#eab308", "#a3a3a3", "#a3a3a3"])
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Lihat detail fitur yang dipakai model"):
                st.dataframe(X, use_container_width=True)
        else:
            st.info("Isi detail rumah di sebelah kiri, lalu klik **Prediksi Harga**.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hd-card">', unsafe_allow_html=True)
    st.markdown("#### Feature Importance Model")
    model = artifact["model"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=artifact["feature_cols"]).sort_values()
        top_feature = imp.idxmax()
        st.caption(f"**{top_feature}** adalah fitur paling berpengaruh terhadap prediksi harga model ini.")
        fig2 = px.bar(imp, orientation="h", color_discrete_sequence=["#eab308"])
        fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="Importance", yaxis_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_about:
    st.markdown('<div class="hd-card">', unsafe_allow_html=True)
    st.markdown("#### Tentang Project")
    st.write(
        "Aplikasi ini memprediksi estimasi harga rumah di wilayah Depok dari data listing properti daring. "
        "Pipeline-nya mencakup pembersihan data mentah (parsing harga & luas, penanganan duplikat dan missing "
        "value), penanganan outlier per kecamatan, feature engineering, perbandingan 5 model regresi, dan "
        "hyperparameter tuning dengan cross-validation."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="hd-stat-label">Model Terbaik</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hd-stat-value">{artifact["model_name"]}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="hd-stat-label">R2 (Test Set)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hd-stat-value">{metrics.get("R2", 0):.3f}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="hd-stat-label">MAPE (Test Set)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hd-stat-value">{metrics.get("MAPE", 0)*100:.1f}%</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Panduan Penggunaan")
    with st.expander("Apa itu aplikasi ini?"):
        st.write("Aplikasi ini memprediksi estimasi harga rumah di wilayah Depok berdasarkan jumlah kamar, "
                 "garasi, luas tanah, luas bangunan, dan kecamatan.")
    with st.expander("Fitur apa saja yang dipakai model?"):
        st.write("Luas Tanah dan Luas Bangunan paling dominan menentukan harga, diikuti Kecamatan dan Garasi. "
                 "Kamar Tidur & Kamar Mandi digabung jadi satu fitur 'Total Kamar' karena keduanya saling "
                 "berkorelasi tinggi, sehingga dipertahankan sebagai satu sinyal saja.")
    with st.expander("Apa itu encoding Kecamatan?"):
        st.write("Tiap kecamatan direpresentasikan sebagai rata-rata harga historis listing di kecamatan "
                 "tersebut, karena lokasi sangat memengaruhi harga rumah di Depok.")
    with st.expander("Apa arti 'Rentang Estimasi'?"):
        st.write("Rentang itu dihitung dari prediksi harga ± rata-rata kesalahan model (MAE) di data test, "
                 "jadi gambaran seberapa jauh prediksi bisa meleset dari harga asli.")
    with st.expander("Seberapa akurat model ini?"):
        st.write("Model dievaluasi dengan 5-Fold Cross Validation di data latih, lalu diuji sekali di data "
                 "test yang belum pernah dilihat model sama sekali, pakai metrik MAE, MAPE, dan R2.")
    st.markdown("</div>", unsafe_allow_html=True)