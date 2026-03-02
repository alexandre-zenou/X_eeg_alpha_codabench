from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.signal import welch

import mne
from moabb.datasets import Rodrigues2017


PHASE = "dev_phase"
INPUT_DIR = Path(PHASE) / "input_data"
REF_DIR = Path(PHASE) / "reference_data"


def make_csv(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(filepath, index=False)


def bandpower_welch(x, sf, fmin=8.0, fmax=12.0):
    freqs, psd = welch(x, fs=sf, nperseg=min(256, len(x)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def features_from_window(window, sf):
    # window shape: (n_channels, n_times)
    row = []
    for ch in window:
        row.append(float(np.mean(ch)))
        row.append(float(np.std(ch)))
        row.append(bandpower_welch(ch, sf, 8.0, 12.0))  # alpha
    return row


def iter_windows(X, sf, win_sec=2.0, max_windows=200, start_samp=0, stop_samp=None):
    win_samp = int(win_sec * sf)
    if stop_samp is None:
        stop_samp = X.shape[1]
    n_times = stop_samp - start_samp
    n_win_total = n_times // win_samp
    n_win = min(n_win_total, max_windows)

    for i in range(n_win):
        a = start_samp + i * win_samp
        b = a + win_samp
        yield X[:, a:b]


def raw_to_examples_event_or_fallback(raw, subject, win_sec=2.0, max_windows_per_class=200):
    """
    Essaie de construire un problème binaire (0/1) :
    - Priorité: utiliser events/annotations si possible
    - Fallback: première moitié du signal = classe 0, seconde moitié = classe 1
    """
    raw = raw.copy().load_data()
    picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, misc=False)
    if len(picks) == 0:
        return [], [], []

    sf = float(raw.info["sfreq"])
    X = raw.get_data(picks=picks)

    X_rows, y_rows, meta_rows = [], [], []

    # 1) TRY events from annotations
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # event_id is dict like {"eyes_open": 1, "eyes_closed": 2, ...} depending on dataset
        if isinstance(event_id, dict) and len(event_id) >= 2:
            # Take two most frequent event types
            # Build epochs around events (fixed length)
            event_keys = list(event_id.keys())

            # create epochs of fixed length starting at each event
            # tmin=0, tmax=win_sec -> exactly one window per event
            epochs = mne.Epochs(
                raw, events, event_id=event_id, tmin=0.0, tmax=win_sec,
                picks=picks, baseline=None, preload=True, verbose=False
            )
            # map first two event types to 0/1
            labels = epochs.events[:, 2]  # numeric event codes
            uniq = np.unique(labels)
            if len(uniq) >= 2:
                code0, code1 = uniq[0], uniq[1]
                data_ep = epochs.get_data()  # (n_epochs, n_channels, n_times)
                # limit windows per class
                c0 = c1 = 0
                for ep, code in zip(data_ep, labels):
                    if code == code0 and c0 < max_windows_per_class:
                        X_rows.append(features_from_window(ep, sf))
                        y_rows.append(0)
                        meta_rows.append({"subject": subject})
                        c0 += 1
                    elif code == code1 and c1 < max_windows_per_class:
                        X_rows.append(features_from_window(ep, sf))
                        y_rows.append(1)
                        meta_rows.append({"subject": subject})
                        c1 += 1
                    if c0 >= max_windows_per_class and c1 >= max_windows_per_class:
                        break

                if len(X_rows) > 0 and len(set(y_rows)) == 2:
                    return X_rows, y_rows, meta_rows
    except Exception:
        pass

    # 2) FALLBACK: time split
    mid = X.shape[1] // 2

    # class 0 = first half
    for window in iter_windows(X, sf, win_sec=win_sec, max_windows=max_windows_per_class, start_samp=0, stop_samp=mid):
        X_rows.append(features_from_window(window, sf))
        y_rows.append(0)
        meta_rows.append({"subject": subject})

    # class 1 = second half
    for window in iter_windows(X, sf, win_sec=win_sec, max_windows=max_windows_per_class, start_samp=mid, stop_samp=X.shape[1]):
        X_rows.append(features_from_window(window, sf))
        y_rows.append(1)
        meta_rows.append({"subject": subject})

    return X_rows, y_rows, meta_rows


def main(seed=42):
    rng = np.random.RandomState(seed)

    train_dir = INPUT_DIR / "train"
    test_dir = INPUT_DIR / "test"
    private_dir = INPUT_DIR / "private_test"
    for d in [train_dir, test_dir, private_dir, REF_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    dataset = Rodrigues2017()
    subjects = dataset.subject_list
    data = dataset.get_data(subjects=subjects)

    X_all, y_all, meta_all = [], [], []

    for subj in data:
        for sess in data[subj]:
            for run, raw in data[subj][sess].items():
                X_rows, y_rows, meta_rows = raw_to_examples_event_or_fallback(
                    raw=raw,
                    subject=subj,
                    win_sec=2.0,
                    max_windows_per_class=200,
                )
                X_all.extend(X_rows)
                y_all.extend(y_rows)
                meta_all.extend(meta_rows)

    X_all = np.asarray(X_all)
    y_all = np.asarray(y_all)
    meta_df = pd.DataFrame(meta_all)

    if len(X_all) == 0:
        raise RuntimeError("No examples were generated. Check raw content / picks.")

    # split par sujets
    unique_subjects = meta_df["subject"].unique()
    subj_train, subj_tmp = train_test_split(unique_subjects, test_size=0.30, random_state=rng)
    subj_test, subj_private = train_test_split(subj_tmp, test_size=0.50, random_state=rng)

    def mask(subjs):
        return meta_df["subject"].isin(subjs).to_numpy()

    m_train = mask(subj_train)
    m_test = mask(subj_test)
    m_private = mask(subj_private)

    X_train, y_train = X_all[m_train], y_all[m_train]
    X_test, y_test = X_all[m_test], y_all[m_test]
    X_private, y_private = X_all[m_private], y_all[m_private]

    make_csv(X_train, train_dir / "train_features.csv")
    make_csv(pd.DataFrame({"label": y_train}), train_dir / "train_labels.csv")

    make_csv(X_test, test_dir / "test_features.csv")
    make_csv(X_private, private_dir / "private_test_features.csv")

    make_csv(pd.DataFrame({"label": y_test}), REF_DIR / "test_labels.csv")
    make_csv(pd.DataFrame({"label": y_private}), REF_DIR / "private_test_labels.csv")

    meta_df.to_csv(Path(PHASE) / "meta.csv", index=False)

    print("OK: wrote dev_phase/input_data and dev_phase/reference_data")
    print("Shapes:", X_train.shape, X_test.shape, X_private.shape)
    print("Class balance train:", np.mean(y_train), "test:", np.mean(y_test), "private:", np.mean(y_private))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create EEG dataset splits for Codabench (events if available, else time-split fallback)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(seed=args.seed)