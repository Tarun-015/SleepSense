
import streamlit as st
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import welch
import tempfile
import os

st.set_page_config(
    page_title="EEG Signal Explorer | SleepSense",
    page_icon="🧠",
    layout="wide"
)

# ---------------- PAGE DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8F7FF, #F1F7FF);
}
h1, h2, h3 {
    color: #303C70 !important;
}
.section-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #E7EAF5;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("EEG Signal Explorer")
st.caption("Explore, preprocess and analyse electroencephalogram recordings.")

st.divider()

# ---------------- FILE UPLOAD ----------------

st.subheader("Upload EEG Recording")

uploaded_file = st.file_uploader(
    "Upload a PSG recording (.edf)",
    type=["edf"],
    help="Upload a Sleep-EDF PSG recording, for example SC4001E0-PSG.edf."
)

if uploaded_file is None:
    st.info(
        "Upload an EDF recording to begin exploring EEG signals."
    )

    st.markdown("""
    **What you can explore here:**
    - EEG channel information
    - Raw brainwave signals
    - Filtered EEG signals
    - Frequency-domain analysis
    - Delta, Theta, Alpha, Sigma and Beta bands
    """)

    st.stop()

# ---------------- LOAD EDF ----------------

@st.cache_data(show_spinner=False)
def load_edf(file_bytes):

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".edf",
            delete=False
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        raw = mne.io.read_raw_edf(
            temp_path,
            preload=True,
            verbose="ERROR"
        )

        data = {
            "data": raw.get_data(),
            "channel_names": raw.ch_names,
            "sfreq": raw.info["sfreq"],
            "duration": raw.times[-1],
            "n_samples": raw.n_times
        }

        return data

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


try:
    with st.spinner("Loading EEG recording..."):
        recording = load_edf(uploaded_file.getvalue())

except Exception as e:
    st.error(f"Unable to read EDF recording: {e}")
    st.stop()


# ---------------- RECORDING INFORMATION ----------------

st.success("EEG recording loaded successfully.")

eeg_channels = [
    channel for channel in recording["channel_names"]
    if "EEG" in channel.upper()
]

if not eeg_channels:
    st.warning(
        "No channel with 'EEG' in its name was found. "
        "Please upload a compatible EEG recording."
    )
    st.stop()

sfreq = recording["sfreq"]
duration = recording["duration"]

st.subheader("Recording Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("EEG Channels", len(eeg_channels))
c2.metric("Sampling Frequency", f"{sfreq:.1f} Hz")
c3.metric("Duration", f"{duration / 60:.1f} min")
c4.metric("Total Samples", f"{recording['n_samples']:,}")

st.write("**Detected EEG channels:**", ", ".join(eeg_channels))

st.divider()

# ---------------- SIGNAL SETTINGS ----------------

st.subheader("Signal Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    selected_channel = st.selectbox(
        "Select EEG Channel",
        eeg_channels
    )

with col2:
    apply_filter = st.checkbox(
        "Apply Bandpass Filter",
        value=True
    )

with col3:
    segment_duration = st.selectbox(
        "Segment Duration",
        [10, 15, 30],
        index=2
    )

max_start = max(0.0, duration - segment_duration)

start_time = st.slider(
    "Select Start Time (seconds)",
    min_value=0.0,
    max_value=float(max_start),
    value=0.0,
    step=1.0
)

# ---------------- SIGNAL EXTRACTION ----------------

channel_index = recording["channel_names"].index(
    selected_channel
)

signal = recording["data"][channel_index].copy()

if apply_filter:

    signal_raw = mne.filter.filter_data(
        signal,
        sfreq=sfreq,
        l_freq=0.3,
        h_freq=min(35.0, sfreq / 2 - 1),
        verbose="ERROR"
    )

    signal_label = "Filtered EEG (0.3–35 Hz)"

else:
    signal_raw = signal
    signal_label = "Raw EEG Signal"


start_sample = int(start_time * sfreq)
end_sample = int((start_time + segment_duration) * sfreq)

segment = signal_raw[start_sample:end_sample]

time_axis = (
    np.arange(len(segment)) / sfreq
) + start_time

segment_uv = segment * 1e6

# ---------------- EEG WAVEFORM ----------------

st.subheader("EEG Waveform")

fig, ax = plt.subplots(figsize=(13, 4))

ax.plot(
    time_axis,
    segment_uv,
    linewidth=0.65
)

ax.set_title(
    f"{selected_channel} | {signal_label}"
)

ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude (µV)")

ax.grid(alpha=0.2)

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Signal amplitude is displayed in microvolts (µV). "
    "MNE stores EEG data in volts internally."
)

st.divider()

# ---------------- FREQUENCY ANALYSIS ----------------

st.subheader("Frequency-Domain Analysis")

if len(segment) < 2:
    st.warning("Not enough samples for frequency analysis.")
    st.stop()

frequencies, psd = welch(
    segment,
    fs=sfreq,
    nperseg=min(len(segment), int(sfreq * 4))
)

fig, ax = plt.subplots(figsize=(13, 4))

ax.plot(
    frequencies,
    10 * np.log10(psd + 1e-20),
    linewidth=1
)

ax.set_xlim(0, min(35, sfreq / 2))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power Spectral Density (dB/Hz)")
ax.set_title("EEG Power Spectral Density")
ax.grid(alpha=0.2)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------- EEG BANDS ----------------

st.subheader("EEG Frequency Bands")

bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Sigma": (11, 16),
    "Beta": (13, 30)
}

band_results = []

total_power = np.trapezoid(
    psd[(frequencies >= 0.5) & (frequencies <= 30)],
    frequencies[(frequencies >= 0.5) & (frequencies <= 30)]
)

for band_name, (low, high) in bands.items():

    mask = (
        (frequencies >= low) &
        (frequencies <= high)
    )

    if mask.sum() >= 2:

        power = np.trapezoid(
            psd[mask],
            frequencies[mask]
        )

        relative_power = (
            power / total_power * 100
            if total_power > 0 else 0
        )

    else:
        power = 0
        relative_power = 0

    band_results.append({
        "Frequency Band": band_name,
        "Frequency Range (Hz)": f"{low}–{high}",
        "Absolute Power (V²)": power,
        "Relative Power (%)": relative_power
    })


band_df = pd.DataFrame(band_results)

st.dataframe(
    band_df.style.format({
        "Absolute Power (V²)": "{:.3e}",
        "Relative Power (%)": "{:.2f}"
    }),
    use_container_width=True,
    hide_index=True
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(
    band_df["Frequency Band"],
    band_df["Relative Power (%)"]
)

ax.set_title("Relative EEG Band Power")
ax.set_ylabel("Relative Power (%)")
ax.set_xlabel("Frequency Band")

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "SleepSense | EEG Signal Explorer | "
    "Research and educational use only."
)
