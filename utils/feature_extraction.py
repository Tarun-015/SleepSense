
import numpy as np
import pandas as pd
from scipy.signal import welch


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (11, 16),
    "beta": (13, 30)
}


def extract_time_features(signal):
    signal = np.asarray(signal)

    activity = np.var(signal)

    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)

    mobility = (
        np.sqrt(np.var(first_derivative) / activity)
        if activity > 0 else 0
    )

    complexity = (
        np.sqrt(
            np.var(second_derivative) /
            np.var(first_derivative)
        ) / mobility
        if np.var(first_derivative) > 0 and mobility > 0
        else 0
    )

    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity
    }


def extract_frequency_features(signal, sfreq):

    frequencies, psd = welch(
        signal,
        fs=sfreq,
        nperseg=min(len(signal), int(4 * sfreq))
    )

    total_mask = (
        (frequencies >= 0.5) &
        (frequencies <= 35)
    )

    total_power = np.trapezoid(
        psd[total_mask],
        frequencies[total_mask]
    )

    features = {}

    for band, (low, high) in BANDS.items():

        mask = (
            (frequencies >= low) &
            (frequencies <= high)
        )

        power = (
            np.trapezoid(psd[mask], frequencies[mask])
            if np.sum(mask) >= 2 else 0
        )

        features[f"{band}_power"] = power

        features[f"{band}_relative_power"] = (
            power / total_power
            if total_power > 0 else 0
        )

    return features


def extract_epoch_features(epoch, sfreq, channel_names):

    epoch_features = {}

    for channel_index, channel_name in enumerate(channel_names):

        signal = epoch[channel_index]

        time_features = extract_time_features(signal)

        frequency_features = extract_frequency_features(
            signal, sfreq
        )

        for name, value in time_features.items():
            epoch_features[f"{channel_name}_{name}"] = value

        for name, value in frequency_features.items():
            epoch_features[f"{channel_name}_{name}"] = value

    return epoch_features


def extract_recording_features(epochs, sfreq, channel_names):

    features_list = []

    for epoch in epochs:

        features = extract_epoch_features(
            epoch,
            sfreq,
            channel_names
        )

        features_list.append(features)

    return pd.DataFrame(features_list)
