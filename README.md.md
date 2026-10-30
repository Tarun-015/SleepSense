# SleepSense: EEG-Based Sleep Stage Classification

SleepSense is an educational/research prototype built with Python and
Streamlit. It explores EEG recordings and uses a classical
machine-learning model to classify consecutive 30-second EEG epochs into
Wake (W), N1, N2, N3 and REM.

> **Disclaimer:** This is not a medical device, diagnostic system or
> substitute for expert polysomnography interpretation. Predictions are
> not clinically validated.

## Project overview

-   Dataset: PhysioNet Sleep-EDF Expanded, Sleep Cassette subset.
-   Input: EDF (`.edf`) EEG recordings.
-   Model channels: `EEG Fpz-Cz` and `EEG Pz-Oz`.
-   Filtering: 0.3--35 Hz band-pass.
-   Epoch length: 30 seconds.
-   Features: 32 ordered time- and frequency-domain features.
-   Selected model: Baseline Support Vector Machine (SVM).
-   Interface: Streamlit.

## Intended user journey

1.  **Overview:** Understand the project, supported input and
    limitations.
2.  **Upload:** Upload an EDF file and inspect recording metadata and
    detected channels.
3.  **Signal Explorer:** Inspect EEG waveforms, filtered signals, Welch
    PSD and band powers.
4.  **Prediction:** Prepare the required channels, divide the recording
    into 30-second epochs, extract model features and run SVM inference.
5.  **Results:** Review predicted-stage distribution, hypnogram-style
    timeline, stage totals and epoch-level predictions.
6.  **Model Insights:** Review held-out test metrics, confusion matrix,
    per-stage results and feature importance.
7.  **Export:** Download epoch-level predictions as CSV.

Recommended navigation: Overview, New Analysis, Results and Model
Insights. The current draft uses separate Streamlit pages for EEG
Explorer, Sleep Prediction, Model Performance and Feature Analysis.
Shared session state should preserve the uploaded recording and results
across the workflow.

## Dataset and labels

Six Sleep Cassette participants were used in the multi-subject work:
`SC4001`, `SC4002`, `SC4011`, `SC4012`, `SC4021` and `SC4022`. The
combined feature dataset contained 16,688 epochs and 36 columns at that
analysis stage.

R&K annotation mapping:

  Original annotation       Project label
  ------------------------- ---------------
  Wake                      W
  Stage 1                   N1
  Stage 2                   N2
  Stage 3 + Stage 4         N3
  REM                       REM
  Movement time / Unknown   Excluded

Approximate class distribution: W 67.9%, N2 16.7%, REM 6.2%, N3 5.6%, N1
3.6%. This imbalance makes accuracy alone insufficient for evaluating
performance.

## Signal processing and features

The documented pipeline uses the two EEG channels above, a 0.3--35 Hz
band-pass filter and consecutive non-overlapping 30-second epochs.

Feature families: - Time domain: mean, standard deviation, RMS, Hjorth
activity, mobility and complexity. - Frequency domain: Welch power
spectral density and absolute/relative band powers.

  Band      Frequency range
  ------- -----------------
  Delta           0.5--4 Hz
  Theta             4--8 Hz
  Alpha            8--13 Hz
  Sigma           11--16 Hz
  Beta            13--30 Hz

The serialized model expects 32 ordered features. **Critical open
validation:** compare the Streamlit feature extractor with the original
Colab training notebook. Matching the feature count does not establish
equivalence. Verify names, order, formulas, PSD integration, channel
order, sampling assumptions, filtering and epoch boundaries before
relying on uploaded-recording predictions.

## Model development and results

Classical models compared: Baseline SVM, Tuned SVM and Random Forest.
The current app uses Baseline SVM because it had the strongest Macro F1
and balanced accuracy in the final held-out comparison. Tuned SVM had
the highest accuracy and Cohen's kappa.

  Model             Accuracy   Balanced accuracy   Macro F1   Cohen's kappa
  --------------- ---------- ------------------- ---------- ---------------
  Baseline SVM        0.8894              0.7609     0.7272          0.7870
  Tuned SVM           0.9032              0.7195     0.7153          0.8084
  Random Forest       0.9007              0.6854     0.6897          0.8002

The final workflow used four participants for development and two
held-out participants for final evaluation. This subject-wise split is
intended to reduce subject leakage, but two test participants are a
small sample. Treat results as preliminary, not broad population
validation.

## Project structure

``` text
SleepSense/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── sleepsense_final_model.pkl
├── data/
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── prediction.py
├── pages/
│   ├── 1_EEG_Explorer.py
│   ├── 2_Sleep_Prediction.py
│   ├── 3_Model_Performance.py
│   └── 4_Feature_Analysis.py
└── assets/
```

  -------------------------------------------------------------------------
  Component                             Responsibility
  ------------------------------------- -----------------------------------
  `app.py`                              Landing page, global configuration
                                        and navigation

  `pages/1_EEG_Explorer.py`             EDF inspection, waveform and
                                        frequency visualization

  `pages/2_Sleep_Prediction.py`         Preprocessing, epoch prediction,
                                        results and CSV export

  `pages/3_Model_Performance.py`        Held-out metrics and model
                                        evaluation visuals

  `pages/4_Feature_Analysis.py`         Permutation importance and feature
                                        explanations

  `utils/preprocessing.py`              Channel selection, filtering and
                                        epoch preparation

  `utils/feature_extraction.py`         Model feature calculations

  `utils/prediction.py`                 Model loading, schema checks and
                                        inference

  `models/sleepsense_final_model.pkl`   Serialized model package and
                                        metadata
  -------------------------------------------------------------------------

The model package was previously inspected as Baseline SVM with 32
feature columns and five classes. It is expected to include the
estimator, model name, ordered feature list and class labels.

## Installation (Windows PowerShell)

Assumes project folder `D:\SleepSense` and virtual environment `venv`:

``` powershell
cd D:\SleepSense
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation:

``` powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

Local address: `http://localhost:8501`

Check dependencies and imports:

``` powershell
python -m pip check
python -c "import streamlit, pandas, numpy, scipy, sklearn, mne, matplotlib, seaborn, joblib; print('All imports successful')"
```

Inspect model metadata:

``` powershell
python -c "import joblib; m=joblib.load('models/sleepsense_final_model.pkl'); print('Model:', m['model_name']); print('Features:', len(m['feature_columns'])); print('Classes:', m['classes'])"
```

Dependencies in the project requirements include Streamlit, pandas,
NumPy, SciPy, scikit-learn, MNE, Matplotlib, Seaborn, joblib and
pyEDFlib. Pin the working versions after the integrated dry run.

## Expected outputs

-   EEG waveform plots for both channels.
-   Frequency spectrum / Welch PSD.
-   Band-power summaries.
-   Predicted class for each 30-second epoch.
-   Predicted-stage distribution and hypnogram-style timeline.
-   Epoch-level prediction table.
-   Downloadable CSV.
-   Static model-performance and feature-importance views.

Do not show accuracy for a newly uploaded recording unless matching
reference annotations are loaded and correctly aligned. A
model-predicted hypnogram is not expert ground truth.

## Limitations and risks

1.  Small participant sample: six participants in multi-subject analysis
    and two held-out participants.
2.  Class imbalance: Wake dominates; N1 is under-represented.
3.  Limited external validation; no broad clinical validation.
4.  Uploaded EDF files may differ in channel names, sampling frequency,
    annotations or signal quality.
5.  Consecutive epochs from recording start may not align with annotated
    sleep epochs unless explicitly aligned.
6.  Feature-extraction mismatch between training and inference can
    invalidate predictions even if the model loads.
7.  Serialized scikit-learn models may require compatible library
    versions.
8.  EEG recordings are sensitive personal data; do not commit raw
    recordings or upload them to unapproved services.
9.  Permutation importance is descriptive, not evidence of biological
    causation.

## Validation checklist

-   [ ] Home page and all navigation links work.
-   [ ] Known compatible EDF uploads successfully.
-   [ ] Required channels are detected and ordered correctly.
-   [ ] Missing channels, corrupt files and short recordings show useful
    errors.
-   [ ] Sampling frequency, duration, filtering and epoch boundaries are
    correct.
-   [ ] All 32 feature names, order and calculations match the training
    notebook.
-   [ ] Feature values match a known training-notebook epoch within a
    defined tolerance.
-   [ ] Predictions contain only W, N1, N2, N3 and REM.
-   [ ] Stage counts reconcile with the prediction table.
-   [ ] CSV export is correct.
-   [ ] Metrics are labelled as held-out test performance.
-   [ ] No clinical claims are made.

## Roadmap

1.  Redesign the interface with a light neuroscience aesthetic and
    consistent navigation.
2.  Build a guided upload → explore → predict → results workflow.
3.  Preserve uploaded data and results using Streamlit session state.
4.  Verify training/inference feature parity.
5.  Test on a known Sleep-EDF recording and document expected output.
6.  Add robust EDF validation and error handling.
7.  Pin dependencies and update this documentation when the pipeline
    changes.

## Dataset citation

The dataset is PhysioNet Sleep-EDF Expanded, Sleep Cassette subset. Use
the official PhysioNet record and associated publication for the exact
bibliographic citation, license and acknowledgement requirements before
formal publication or redistribution. The precise citation has not been
independently verified in this README.
