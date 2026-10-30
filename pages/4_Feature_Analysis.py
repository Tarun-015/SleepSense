
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Feature Analysis | SleepSense",
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


# ---------------- HEADER ----------------

st.title("EEG Feature Analysis")

st.caption(
    "Understanding the signal characteristics used by the "
    "SleepSense sleep-stage classification model."
)

st.divider()


# ---------------- INTRODUCTION ----------------

st.subheader("Feature Engineering Overview")

st.markdown("""
SleepSense extracts 32 features from two EEG channels:

- EEG Fpz-Cz
- EEG Pz-Oz

The features represent two broad categories:

1. **Time-domain features:** describe signal amplitude and variation.
2. **Frequency-domain features:** describe the distribution of signal power
   across different EEG frequency bands.
""")

col1, col2, col3 = st.columns(3)

col1.metric("EEG Channels", "2")
col2.metric("Features per Channel", "16")
col3.metric("Total Features", "32")

st.divider()


# ---------------- FEATURE GROUPS ----------------

st.subheader("Feature Categories")

time_features = pd.DataFrame({
    "Feature": [
        "Mean",
        "Standard Deviation",
        "RMS",
        "Hjorth Activity",
        "Hjorth Mobility",
        "Hjorth Complexity"
    ],
    "Interpretation": [
        "Average signal amplitude",
        "Signal variability",
        "Overall signal magnitude",
        "Variance of the EEG signal",
        "Relative frequency or rate of signal change",
        "Complexity of signal variation"
    ]
})

frequency_features = pd.DataFrame({
    "Feature": [
        "Delta Power",
        "Theta Power",
        "Alpha Power",
        "Sigma Power",
        "Beta Power",
        "Relative Band Power"
    ],
    "Frequency Range": [
        "0.5–4 Hz",
        "4–8 Hz",
        "8–13 Hz",
        "11–16 Hz",
        "13–30 Hz",
        "Proportion of total power"
    ],
    "Interpretation": [
        "Slow-wave activity",
        "Low-frequency rhythmic activity",
        "Alpha-band activity",
        "Sleep-spindle-associated frequency range",
        "Higher-frequency EEG activity",
        "Band power normalized by total power"
    ]
})

tab1, tab2 = st.tabs([
    "Time-Domain Features",
    "Frequency-Domain Features"
])

with tab1:
    st.dataframe(
        time_features,
        use_container_width=True,
        hide_index=True
    )

with tab2:
    st.dataframe(
        frequency_features,
        use_container_width=True,
        hide_index=True
    )

st.divider()


# ---------------- PERMUTATION IMPORTANCE ----------------

st.subheader("Permutation Feature Importance")

st.markdown("""
Permutation importance measures how much model performance decreases
when the values of one feature are randomly shuffled.

A larger decrease indicates that the model relies more heavily on
that feature for its predictions.
""")

importance_df = pd.DataFrame({
    "Feature": [
        "EEG Pz-Oz_hjorth_mobility",
        "EEG Fpz-Cz_hjorth_complexity",
        "EEG Pz-Oz_delta_power",
        "EEG Pz-Oz_beta_relative_power",
        "EEG Pz-Oz_delta_relative_power",
        "EEG Pz-Oz_alpha_relative_power",
        "EEG Fpz-Cz_theta_relative_power",
        "EEG Fpz-Cz_sigma_relative_power",
        "EEG Pz-Oz_beta_power",
        "EEG Fpz-Cz_delta_relative_power",
        "EEG Pz-Oz_sigma_relative_power",
        "EEG Pz-Oz_theta_power",
        "EEG Pz-Oz_sigma_power",
        "EEG Pz-Oz_alpha_power",
        "EEG Fpz-Cz_hjorth_mobility"
    ],
    "Importance": [
        0.12133,
        0.07618,
        0.06719,
        0.05902,
        0.05548,
        0.04210,
        0.04149,
        0.04128,
        0.04070,
        0.03210,
        0.03170,
        0.02703,
        0.02580,
        0.02382,
        0.02183
    ],
    "Std": [
        0.00752,
        0.00478,
        0.00559,
        0.00671,
        0.00772,
        0.00552,
        0.00156,
        0.00365,
        0.00680,
        0.00225,
        0.00412,
        0.00636,
        0.00258,
        0.00332,
        0.00266
    ]
})

importance_df["Channel"] = importance_df["Feature"].str.extract(
    r"(EEG [^_]+)"
)

importance_df["Feature Type"] = importance_df["Feature"].apply(
    lambda name:
        "Time Domain"
        if "hjorth" in name
        else "Frequency Domain"
)

importance_df["Feature Name"] = importance_df["Feature"].str.replace(
    r"^EEG (Fpz-Cz|Pz-Oz)_",
    "",
    regex=True
)

st.caption(
    "Top 15 features from the baseline SVM permutation-importance analysis."
)

fig, ax = plt.subplots(figsize=(12, 7))

plot_df = importance_df.sort_values(
    "Importance",
    ascending=True
)

ax.barh(
    plot_df["Feature"],
    plot_df["Importance"],
    xerr=plot_df["Std"],
    capsize=3
)

ax.set_xlabel("Mean Permutation Importance")
ax.set_ylabel("EEG Feature")
ax.set_title("Top 15 Features by Permutation Importance")
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ---------------- FEATURE TABLE ----------------

st.subheader("Feature Importance Ranking")

st.dataframe(
    importance_df[
        [
            "Feature",
            "Channel",
            "Feature Type",
            "Importance",
            "Std"
        ]
    ].style.format({
        "Importance": "{:.5f}",
        "Std": "{:.5f}"
    }),
    use_container_width=True,
    hide_index=True
)

csv_data = importance_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Feature Importance CSV",
    data=csv_data,
    file_name="sleepsense_feature_importance.csv",
    mime="text/csv"
)

st.divider()


# ---------------- FEATURE GROUP ANALYSIS ----------------

st.subheader("Feature Group Analysis")

group_summary = (
    importance_df.groupby("Feature Type")["Importance"]
    .agg(["mean", "sum", "count"])
    .reset_index()
)

group_summary.columns = [
    "Feature Type",
    "Mean Importance",
    "Total Importance",
    "Features in Top 15"
]

st.dataframe(
    group_summary.style.format({
        "Mean Importance": "{:.5f}",
        "Total Importance": "{:.5f}"
    }),
    use_container_width=True,
    hide_index=True
)

channel_summary = (
    importance_df.groupby("Channel")["Importance"]
    .agg(["mean", "sum", "count"])
    .reset_index()
)

channel_summary.columns = [
    "EEG Channel",
    "Mean Importance",
    "Total Importance",
    "Features in Top 15"
]

st.subheader("Channel-Level Feature Contribution")

st.dataframe(
    channel_summary.style.format({
        "Mean Importance": "{:.5f}",
        "Total Importance": "{:.5f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "These summaries describe only the selected top 15 features, "
    "not all 32 features."
)

st.divider()


# ---------------- INTERPRETATION ----------------

st.subheader("What Do These Results Suggest?")

st.markdown("""
**1. Hjorth mobility and complexity**

The high ranking of Hjorth mobility and complexity suggests that
signal dynamics and changes in EEG activity are useful to the model.

**2. Spectral characteristics**

Delta power and relative power features appear prominently.
This suggests that frequency-domain characteristics also contribute
to sleep-stage discrimination.

**3. Channel observations**

Several of the highly ranked features come from Pz-Oz.
This indicates that the model uses information from this channel
in its predictions. It does not prove that Pz-Oz is inherently
better than Fpz-Cz.

**4. Combined feature representation**

The current model uses both signal dynamics and frequency-band
characteristics rather than relying on a single EEG measurement.
""")

st.warning("""
Permutation importance is a measure of predictive dependence, not causation.
Correlated EEG features can share or obscure importance.

These importance values were calculated on the held-out test set.
They are presented here for interpretation only and were not used
to select features or retrain the model.
""")

st.divider()

st.caption(
    "SleepSense | EEG Feature Analysis | Research and educational use only."
)
