
import mne
import numpy as np


EXPECTED_CHANNELS = [
    "EEG Fpz-Cz",
    "EEG Pz-Oz"
]


def load_edf(file_path):

    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose="ERROR"
    )

    return raw


def prepare_eeg(raw):

    available_channels = raw.ch_names

    missing_channels = [
        ch for ch in EXPECTED_CHANNELS
        if ch not in available_channels
    ]

    if missing_channels:
        raise ValueError(
            "Required EEG channels are missing: "
            + ", ".join(missing_channels)
        )

    eeg_raw = raw.copy().pick(
        EXPECTED_CHANNELS
    )

    eeg_raw.filter(
        l_freq=0.3,
        h_freq=35.0,
        method="fir",
        verbose="ERROR"
    )

    return eeg_raw


def create_epochs(eeg_raw, epoch_duration=30):

    sfreq = eeg_raw.info["sfreq"]

    samples_per_epoch = int(epoch_duration * sfreq)

    eeg_data = eeg_raw.get_data()

    total_samples = eeg_data.shape[1]

    number_of_epochs = total_samples // samples_per_epoch

    if number_of_epochs == 0:
        raise ValueError(
            "Recording is too short to create a 30-second epoch."
        )

    usable_samples = number_of_epochs * samples_per_epoch

    eeg_data = eeg_data[:, :usable_samples]

    epochs = eeg_data.reshape(
        len(EXPECTED_CHANNELS),
        number_of_epochs,
        samples_per_epoch
    ).transpose(1, 0, 2)

    return epochs, sfreq, EXPECTED_CHANNELS
