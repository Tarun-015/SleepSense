
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Model Performance | SleepSense",
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

st.title("Model Performance")
st.caption(
    "Evaluation and comparison of classical machine-learning models "
    "for EEG-based sleep-stage classification."
)

st.divider()


# ---------------- EXPERIMENT CONTEXT ----------------

st.subheader("Evaluation Methodology")

st.markdown("""
The final model comparison uses a participant-level development/test split.
Four participants were used for model development, while two participants
were held out for final evaluation.

- Development participants: SC4001, SC4002, SC4011, SC4012
- Held-out test participants: SC4021, SC4022
- Evaluation: subject-independent held-out test
- Primary metrics: Macro F1 and Balanced Accuracy
""")

c1, c2, c3 = st.columns(3)
c1.metric("Development Participants", "4")
c2.metric("Held-out Participants", "2")
c3.metric("Held-out Epochs", "5,559")

st.warning(
    "The held-out test set contains only two participants. "
    "These results are preliminary and should not be interpreted "
    "as evidence of clinical or population-wide generalization."
)

st.divider()


# ---------------- OVERALL COMPARISON ----------------

st.subheader("Overall Model Comparison")

comparison_df = pd.DataFrame({
    "Model": [
        "Baseline SVM",
        "Tuned SVM",
        "Random Forest"
    ],
    "Accuracy": [
        0.8894,
        0.9032,
        0.9007
    ],
    "Balanced Accuracy": [
        0.7609,
        0.7195,
        0.6854
    ],
    "Macro F1": [
        0.7272,
        0.7153,
        0.6897
    ],
    "Cohen's Kappa": [
        0.7870,
        0.8084,
        0.8002
    ]
})

display_df = comparison_df.copy()

metric_columns = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1",
    "Cohen's Kappa"
]

for col in metric_columns:
    display_df[col] = display_df[col].map(
        lambda value: f"{value:.4f}"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "All three models were evaluated on the same held-out participants."
)


# ---------------- KEY METRICS ----------------

st.subheader("Selected Model: Baseline SVM")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

metric_col1.metric("Accuracy", "88.94%")
metric_col2.metric("Balanced Accuracy", "76.09%")
metric_col3.metric("Macro F1", "72.72%")
metric_col4.metric("Cohen's Kappa", "0.7870")

st.info(
    "Baseline SVM is retained as the current preferred model because "
    "it achieved the highest Macro F1 and Balanced Accuracy among "
    "the three evaluated models."
)

st.divider()


# ---------------- METRIC COMPARISON CHART ----------------

st.subheader("Performance Across Models")

selected_metric = st.selectbox(
    "Choose evaluation metric",
    metric_columns,
    index=2
)

fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(
    comparison_df["Model"],
    comparison_df[selected_metric] * 100
)

ax.set_ylabel(f"{selected_metric} (%)")
ax.set_title(f"{selected_metric} Comparison")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ---------------- STAGE-WISE F1 ----------------

st.divider()
st.subheader("Sleep-Stage Classification Performance")

stage_f1 = pd.DataFrame({
    "Sleep Stage": ["W", "N1", "N2", "N3", "REM"],
    "Baseline SVM": [0.9680, 0.2773, 0.8663, 0.7736, 0.7506],
    "Tuned SVM": [0.9795, 0.2686, 0.8556, 0.6824, 0.7906],
    "Random Forest": [0.9747, 0.2016, 0.8511, 0.6667, 0.7546]
})

st.markdown(
    "Macro-level metrics can hide differences between individual "
    "sleep stages. The chart below shows the F1 score for each stage."
)

fig, ax = plt.subplots(figsize=(11, 5))

x = np.arange(len(stage_f1))
width = 0.24

ax.bar(
    x - width,
    stage_f1["Baseline SVM"],
    width,
    label="Baseline SVM"
)

ax.bar(
    x,
    stage_f1["Tuned SVM"],
    width,
    label="Tuned SVM"
)

ax.bar(
    x + width,
    stage_f1["Random Forest"],
    width,
    label="Random Forest"
)

ax.set_xticks(x)
ax.set_xticklabels(stage_f1["Sleep Stage"])
ax.set_ylabel("F1 Score")
ax.set_ylim(0, 1.05)
ax.set_title("F1 Score by Sleep Stage")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.dataframe(
    stage_f1.style.format({
        "Baseline SVM": "{:.4f}",
        "Tuned SVM": "{:.4f}",
        "Random Forest": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.warning(
    "N1 remains the most challenging class. Its low F1 scores indicate "
    "that the current feature-based models struggle to distinguish "
    "N1 reliably."
)


# ---------------- TUNED SVM CONFUSION MATRIX ----------------

st.divider()
st.subheader("Tuned SVM Confusion Matrix")

st.caption(
    "This confusion matrix belongs to the tuned SVM evaluation only. "
    "Rows represent actual labels and columns represent predicted labels."
)

confusion_matrix = np.array([
    [3675, 87, 13, 0, 3],
    [34, 65, 86, 0, 93],
    [2, 35, 847, 36, 27],
    [8, 3, 73, 130, 0],
    [7, 16, 14, 1, 304]
])

labels = ["W", "N1", "N2", "N3", "REM"]

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(
    confusion_matrix,
    interpolation="nearest"
)

ax.set_title("Tuned SVM: Confusion Matrix")
ax.set_xlabel("Predicted Stage")
ax.set_ylabel("Actual Stage")

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

threshold = confusion_matrix.max() / 2

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        ax.text(
            j,
            i,
            str(confusion_matrix[i, j]),
            ha="center",
            va="center",
            color="white" if confusion_matrix[i, j] > threshold else "black"
        )

fig.colorbar(image, ax=ax, label="Number of Epochs")
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ---------------- INTERPRETATION ----------------

st.divider()
st.subheader("Research Interpretation")

st.markdown("""
**1. Wake-stage classification**

All three models perform strongly on Wake (W). However, this class
has the largest representation in the dataset.

**2. N1 classification**

N1 remains difficult to classify. The baseline SVM achieves a
higher N1 F1 score than the tuned SVM and Random Forest.

**3. N2 and N3 classification**

The baseline SVM performs comparatively well on N2 and N3.
The tuned SVM improves some metrics but does not improve overall
macro-level performance.

**4. Model selection**

Hyperparameter tuning improved accuracy and Cohen's Kappa, but
reduced Macro F1 and Balanced Accuracy. This illustrates why
accuracy alone is insufficient for imbalanced sleep-stage data.
""")

st.divider()

st.caption(
    "SleepSense | Model Evaluation | Research and educational use only."
)
