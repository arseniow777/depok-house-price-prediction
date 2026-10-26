import joblib
import numpy as np
import pandas as pd


def load_artifact(path):
    return joblib.load(path)


def encode_kecamatan(kecamatan, kec_target_map):
    if kecamatan in kec_target_map:
        return kec_target_map[kecamatan]
    return float(np.mean(list(kec_target_map.values())))


def build_features(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, kecamatan, artifact):
    kec_enc = encode_kecamatan(kecamatan, artifact["kec_target_map"])
    row = {
        "Garasi": garasi,
        "Luas Tanah": luas_tanah,
        "Luas Bangunan": luas_bangunan,
        "Kecamatan_enc": kec_enc,
        "Total Kamar": kamar_tidur + kamar_mandi,
    }
    df = pd.DataFrame([row])
    return df[artifact["feature_cols"]]


def predict_price(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, kecamatan, artifact):
    X = build_features(kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, kecamatan, artifact)
    pred = artifact["model"].predict(X)[0]
    if artifact.get("use_log_target"):
        pred = np.expm1(pred)
    return max(float(pred), 0.0), X