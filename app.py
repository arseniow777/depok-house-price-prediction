import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.predict import load_artifact, predict_price

st.set_page_config(page_title="Depok House Price Predictor", page_icon="🏠", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.joblib")


@st.cache_resource
def get_artifact():
    return load_artifact(MODEL_PATH)


artifact = get_artifact()

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

with st.sidebar:
    st.session_state.dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)


def theme_colors(dark: bool):
    if dark:
        return dict(bg="#0f0f0f", card="#1a1a1a", text="#fafafa", subtext="#a3a3a3",
                    border="#2a2a2a", accent="#eab308", chart_muted="#525252")
    return dict(bg="#ffffff", card="#fafafa", text="#111111", subtext="#525252",
                border="#e5e5e5", accent="#eab308", chart_muted="#a3a3a3")


c = theme_colors(st.session_state.dark_mode)

st.markdown(f"""
<style>
.stApp {{ background-color: {c['bg']}; color: {c['text']}; }}
section[data-testid="stSidebar"] {{ background-color: {c['card']}; border-right: 1px solid {c['border']}; }}
.hd-card {{
    background-color: {c['card']};
    border: 1px solid {c['border']};
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}}
.hd-title {{ font-weight: 800; font-size: 2.1rem; letter-spacing: -0.02em; color: {c['text']}; }}
.hd-subtitle {{ color: {c['subtext']}; font-size: 0.95rem; }}
.hd-divider {{ border: none; border-top: 2px solid {c['text']}; margin: 0.75rem 0 1.5rem 0; }}
.hd-label {{ color: {c['subtext']}; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }}
.hd-price {{ font-size: 2.3rem; font-weight: 800; color: {c['accent']}; }}
.stButton > button {{
    background-color: {c['accent']};
    color: #111111;
    font-weight: 700;
    border: none;
    border-radius: 6px;
}}
.stButton > button:hover {{ background-color: #ca8a04; color: #111111; }}
h4 {{ color: {c['text']}; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hd-title">DEPOK HOUSE PRICE PREDICTOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hd-subtitle">Estimasi harga rumah di wilayah Depok berdasarkan luas, lokasi, dan fasilitas — '
    'ditenagai model machine learning hasil riset.</div>', unsafe_allow_html=True)
st.markdown('<hr class="hd-divider">', unsafe_allow_html=True)

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
        price, _ = predict_price(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, kecamatan, artifact)

        st.markdown('<div class="hd-label">Estimasi Harga</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="hd-price">Rp {price:,.0f}</div>', unsafe_allow_html=True)
        st.caption(f"Model: {artifact['model_name']}")

        kec_avg = artifact["kec_target_map"][kecamatan]
        depok_avg = float(np.mean(list(artifact["kec_target_map"].values())))
        comp_df = pd.DataFrame({
            "Kategori": ["Prediksi Kamu", f"Rata-rata {kecamatan}", "Rata-rata Depok"],
            "Harga": [price, kec_avg, depok_avg],
        })
        fig = px.bar(comp_df, x="Kategori", y="Harga", color="Kategori",
                     color_discrete_sequence=[c["accent"], c["chart_muted"], c["chart_muted"]])
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font_color=c["text"], margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Isi detail rumah di sebelah kiri, lalu klik **Prediksi Harga**.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="hd-card">', unsafe_allow_html=True)
st.markdown("#### Feature Importance Model")
model = artifact["model"]
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=artifact["feature_cols"]).sort_values()
    fig2 = px.bar(imp, orientation="h", color_discrete_sequence=[c["accent"]])
    fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color=c["text"], xaxis_title="Importance", yaxis_title="", margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("#### Panduan Penggunaan")
with st.expander("Apa itu aplikasi ini?"):
    st.write("Aplikasi ini memprediksi estimasi harga rumah di wilayah Depok berdasarkan jumlah kamar, "
             "garasi, luas tanah, luas bangunan, dan kecamatan, menggunakan model machine learning yang "
             "dilatih dari data listing rumah di Depok.")
with st.expander("Fitur apa saja yang dipakai model?"):
    st.write("Luas Tanah dan Luas Bangunan paling dominan menentukan harga, diikuti Kecamatan dan Garasi. "
             "Kamar Tidur & Kamar Mandi digabung jadi satu fitur 'Total Kamar' karena keduanya saling "
             "berkorelasi tinggi.")
with st.expander("Apa itu encoding Kecamatan?"):
    st.write("Tiap kecamatan direpresentasikan sebagai rata-rata harga historis listing di kecamatan "
             "tersebut, karena lokasi sangat memengaruhi harga rumah di Depok.")
with st.expander("Seberapa akurat model ini?"):
    st.write("Model dievaluasi dengan 5-Fold Cross Validation di data latih, lalu diuji sekali di data "
             "test yang belum pernah dilihat model, menggunakan metrik MAE, MAPE, dan R2.")