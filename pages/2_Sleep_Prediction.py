
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from utils.preprocessing import load_edf, prepare_eeg, create_epochs
from utils.prediction import predict_sleep_stages

st.set_page_config(
    page_title="Sleep Prediction | SleepSense",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8F7FF, #F1F7FF);
}
h1, h2, h3 {
    color: #303C70 !important;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E7EAF5;
    padding: 16px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.title("Sleep Stage Prediction")
st.caption(
    "Classify 30-second EEG segments using the trained SleepSense SVM model."
)

st.divider()

# ---------------- MODEL INFORMATION ----------------

st.subheader("Prediction Model")

col1, col2, col3 = st.columns(3)

col1.metric("Selected Model", "Baseline SVM")
col2.metric("Input Features", "32")
col3.metric("Sleep Stages", "5")

st.markdown("""
The model uses EEG time-domain and frequency-domain features
from the Fpz-Cz and Pz-Oz channels.
""")

st.divider()

# ---------------- EDF UPLOAD ----------------

st.subheader("Upload PSG Recording")

uploaded_file = st.file_uploader(
    "Choose an EEG PSG recording",
    type=["edf"],
    help="Upload a Sleep-EDF PSG file, for example SC4001E0-PSG.edf."
)

if uploaded_file is None:
    st.info("Upload a PSG EDF file to generate sleep-stage predictions.")
    st.stop()

# ---------------- PROCESS RECORDING ----------------

if st.button("Generate Sleep Predictions", type="primary"):

    temp_path = None

    try:
        with st.spinner("Processing EEG recording and generating predictions..."):

            # Save uploaded file temporarily for MNE
            with tempfile.NamedTemporaryFile(
                suffix=".edf",
                delete=False
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            # Load recording
            raw = load_edf(temp_path)

            # Select expected EEG channels and filter
            eeg_raw = prepare_eeg(raw)

            # Create consecutive 30-second epochs
            epochs, sfreq, channel_names = create_epochs(
                eeg_raw,
                epoch_duration=30
            )

            # Predict using the saved SVM model
            results, feature_df = predict_sleep_stages(
                epochs,
                sfreq,
                channel_names
            )

            # Store results for display and download
            st.session_state["sleep_predictions"] = results
            st.session_state["sleep_features"] = feature_df

            st.session_state["prediction_metadata"] = {
                "channels": channel_names,
                "sfreq": sfreq,
                "epochs": len(epochs),
                "duration_minutes": len(epochs) * 30 / 60
            }

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------- DISPLAY RESULTS ----------------

if "sleep_predictions" in st.session_state:

    results = st.session_state["sleep_predictions"]
    metadata = st.session_state["prediction_metadata"]

    st.success("Sleep-stage predictions generated.")

    st.subheader("Recording Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Epochs", f"{metadata['epochs']:,}")
    c2.metric("Duration Analysed", f"{metadata['duration_minutes']:.1f} min")
    c3.metric("Sampling Frequency", f"{metadata['sfreq']:.1f} Hz")
    c4.metric("EEG Channels", len(metadata["channels"]))

    st.caption(
        "Channels: " + ", ".join(metadata["channels"])
    )

    st.divider()

    # Sleep stage distribution
    st.subheader("Predicted Sleep-Stage Distribution")

    stage_order = ["W", "N1", "N2", "N3", "REM"]

    stage_counts = (
        results["Predicted Sleep Stage"]
        .value_counts()
        .reindex(stage_order, fill_value=0)
    )

    total_epochs = len(results)

    distribution = pd.DataFrame({
        "Sleep Stage": stage_counts.index,
        "Epoch Count": stage_counts.values,
        "Percentage": (
            stage_counts.values / total_epochs * 100
        )
    })

    metric_columns = st.columns(5)

    for column, row in zip(metric_columns, distribution.itertuples()):
        column.metric(
            row[1],
            f"{row[3]:.1f}%"
        )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        distribution["Sleep Stage"],
        distribution["Percentage"]
    )

    ax.set_xlabel("Predicted Sleep Stage")
    ax.set_ylabel("Percentage of Epochs")
    ax.set_title("Sleep-Stage Distribution")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Prediction timeline
    st.subheader("Sleep-Stage Timeline")

    timeline = results.copy()
    timeline["Time (minutes)"] = (
        (timeline["Epoch"] - 1) * 30 / 60
    )

    stage_codes = {
        "W": 0,
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4
    }

    timeline["Stage Code"] = (
        timeline["Predicted Sleep Stage"].map(stage_codes)
    )

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.step(
        timeline["Time (minutes)"],
        timeline["Stage Code"],
        where="post",
        linewidth=1.4
    )

    ax.set_yticks(
        list(stage_codes.values()),
        list(stage_codes.keys())
    )

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Predicted Sleep Stage")
    ax.set_title("Predicted Sleep-Stage Timeline")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Detailed predictions
    st.subheader("Epoch-Level Predictions")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # Download results
    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions CSV",
        data=csv_data,
        file_name="sleepsense_predictions.csv",
        mime="text/csv"
    )

    # Feature export
    feature_data = st.session_state["sleep_features"].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Extracted Features CSV",
        data=feature_data,
        file_name="sleepsense_extracted_features.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "SleepSense | Research and educational use only. "
    "Predictions are not a medical diagnosis."
)
