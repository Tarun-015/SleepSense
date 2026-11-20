
import streamlit as st

st.set_page_config(
    page_title="SleepSense | EEG Sleep Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #F8F7FF 0%,
        #F1F7FF 55%,
        #F8FAFF 100%
    );
}

header[data-testid="stHeader"] {
    background: transparent;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

h1, h2, h3 {
    color: #303C70 !important;
}

.hero {
    background: linear-gradient(
        120deg,
        #E9E5FF,
        #E4F2FF
    );
    padding: 42px;
    border-radius: 24px;
    border: 1px solid #DFE4F7;
    margin-bottom: 28px;
}

.hero-tag {
    color: #6C63A9;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #303C70;
    font-size: 43px;
    font-weight: 750;
    margin: 12px 0;
}

.hero-description {
    color: #586586;
    font-size: 17px;
    line-height: 1.7;
    max-width: 780px;
}

.feature-card {
    background: rgba(255,255,255,0.92);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #E5E9F5;
    min-height: 185px;
}

.feature-icon {
    font-size: 27px;
    margin-bottom: 10px;
}

.feature-title {
    font-size: 19px;
    font-weight: 700;
    color: #354276;
    margin-bottom: 8px;
}

.feature-description {
    color: #68738D;
    font-size: 14px;
    line-height: 1.6;
}

.stage-pill {
    background: #FFFFFF;
    border: 1px solid #E0E5F2;
    color: #47547D;
    padding: 11px 18px;
    border-radius: 30px;
    text-align: center;
    font-weight: 650;
}

.info-card {
    background: white;
    border: 1px solid #E5E9F5;
    padding: 22px;
    border-radius: 17px;
}

.small-label {
    color: #858CA5;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HERO ----------------

st.markdown("""
<div class="hero">
    <div class="hero-tag">EEG • Neuroscience • Machine Learning</div>
    <div class="hero-title">SleepSense 🧠</div>
    <div class="hero-description">
        An EEG-based sleep-stage classification platform that explores
        brainwave signals, extracts meaningful signal characteristics
        and applies machine learning to classify sleep stages.
        <br><br>
        Turning electrophysiological signals into interpretable insights.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- OVERVIEW ----------------

st.subheader("Explore SleepSense")

st.markdown(
    "Navigate through the platform's signal-processing, "
    "prediction and model-interpretation modules."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">〰️</div>
        <div class="feature-title">EEG Signal Explorer</div>
        <div class="feature-description">
            Upload EEG recordings, inspect signal waveforms,
            apply preprocessing filters and explore
            frequency-domain characteristics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/1_EEG_Explorer.py",
        label="Explore EEG Signals →",
        icon="〰️"
    )

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🌙</div>
        <div class="feature-title">Sleep Stage Prediction</div>
        <div class="feature-description">
            Generate sleep-stage predictions from EEG epochs
            using the trained Support Vector Machine model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/2_Sleep_Prediction.py",
        label="Predict Sleep Stages →",
        icon="🌙"
    )

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Model Performance</div>
        <div class="feature-description">
            Compare Random Forest, baseline SVM and tuned SVM
            using held-out participant evaluation results.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/3_Model_Performance.py",
        label="View Model Performance →",
        icon="📊"
    )

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">Feature Analysis</div>
        <div class="feature-description">
            Understand EEG time-domain and frequency-domain
            features and explore permutation importance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/4_Feature_Analysis.py",
        label="Explore EEG Features →",
        icon="🔬"
    )


# ---------------- SLEEP STAGES ----------------

st.divider()

st.subheader("Understanding Sleep Stages")

st.markdown(
    "The classification system considers five sleep-stage categories."
)

stage_col1, stage_col2, stage_col3, stage_col4, stage_col5 = st.columns(5)

stages = [
    (stage_col1, "W", "Wake"),
    (stage_col2, "N1", "Light Sleep"),
    (stage_col3, "N2", "Intermediate Sleep"),
    (stage_col4, "N3", "Deep Sleep"),
    (stage_col5, "REM", "Rapid Eye Movement")
]

for column, code, description in stages:
    with column:
        st.markdown(
            f"""
            <div class="stage-pill">
                <div style="font-size:21px">{code}</div>
                <div style="font-size:12px;font-weight:400">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------- PROJECT DETAILS ----------------

st.divider()

st.subheader("Inside the Research")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="small-label">Dataset</div>
        <h3>Sleep-EDF Expanded</h3>
        <p>
            Publicly available polysomnography recordings
            and sleep-stage annotations from PhysioNet.
        </p>
        <div class="small-label">Signal Processing</div>
        <p>
            EEG filtering, 30-second epoching, Welch PSD,
            Hjorth parameters and spectral band-power extraction.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="small-label">Machine Learning</div>
        <h3>Classical ML Models</h3>
        <p>
            Random Forest and Support Vector Machine models,
            including SVM hyperparameter optimization.
        </p>
        <div class="small-label">Evaluation</div>
        <p>
            Subject-independent evaluation using Macro F1,
            balanced accuracy, Cohen's Kappa and confusion matrices.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div style="text-align:center; padding:18px; color:#858CA5;">
    <b style="color:#53618A;">SleepSense</b>
    <br>
    EEG-Based Sleep Stage Classification
    <br><br>
    Research and educational project · Not intended for medical diagnosis
</div>
""", unsafe_allow_html=True)
