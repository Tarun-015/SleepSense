
import joblib
import pandas as pd
from pathlib import Path

from utils.feature_extraction import extract_recording_features


MODEL_PATH = Path("models/sleepsense_final_model.pkl")


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_sleep_stages(epochs, sfreq, channel_names):

    model_package = load_model()

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]

    features_df = extract_recording_features(
        epochs,
        sfreq,
        channel_names
    )

    # Ensure exact feature order expected by the trained model.
    missing_features = [
        col for col in feature_columns
        if col not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing_features)
        )

    features_df = features_df[feature_columns]

    predictions = model.predict(features_df)

    results = pd.DataFrame({
        "Epoch": range(1, len(predictions) + 1),
        "Predicted Sleep Stage": predictions
    })

    return results, features_df
