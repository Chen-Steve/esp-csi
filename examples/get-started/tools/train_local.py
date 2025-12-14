# Local training script (no Colab dependencies)

import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# LOCAL PATHS - adjust these
DATA_DIR = "./data"  # folder containing no_person/ and person_present/ subfolders
MODEL_PATH = "./csi_model_local.pkl"

LABELS = {0: "no_person", 1: "person_present"}
WINDOW_SEC = 2.0
STEP_SEC = 0.5
PACKET_RATE = 20


def parse_csi(csi_str):
    raw = json.loads(csi_str)
    n = len(raw) // 2
    return np.array([complex(raw[i*2+1], raw[i*2]) for i in range(n)], dtype=np.complex64)


def load_csv(filepath):
    df = pd.read_csv(filepath)
    entries = []
    for _, row in df.iterrows():
        try:
            entries.append({
                'csi': parse_csi(row['data']),
                'agc': row.get('agc_gain', 0),
                'fft': row.get('fft_gain', 0)
            })
        except Exception as e:
            print("Row parse error:", e)
            continue
    return entries


def load_all_data():
    data = []
    for label, dirname in LABELS.items():
        path = Path(DATA_DIR) / dirname
        if not path.exists():
            print(f"Warning: {path} not found")
            continue
        files = list(path.glob("*.csv"))
        print(f"Loading {len(files)} files from {dirname}")
        for f in files:
            entries = load_csv(f)
            if entries:
                data.append((entries, label))
    return data


def bandpass(data, fs, low=0.1, high=5.0):
    if len(data) < 20:
        return data
    b, a = signal.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return signal.filtfilt(b, a, data)


def preprocess(entries):
    n_pkt = len(entries)
    n_sc = len(entries[0]['csi'])
    amp = np.zeros((n_pkt, n_sc))
    phase = np.zeros((n_pkt, n_sc))

    for i, e in enumerate(entries):
        csi = e['csi']
        gain_db = e['agc'] + e['fft']
        
        a = np.abs(csi)
        if gain_db != 0:
            a = a / (10 ** (gain_db / 20))
        amp[i, :len(a)] = a
        phase[i, :len(csi)] = np.angle(csi)

    if n_pkt > 20:
        for j in range(n_sc):
            amp[:, j] = bandpass(amp[:, j], PACKET_RATE)
    
    return amp, phase


def extract_features(amp, phase):
    f = {}
    
    # Amplitude features
    f['mean'] = np.mean(amp)
    f['std'] = np.std(amp)
    f['var'] = np.var(amp)
    f['range'] = np.max(amp) - np.min(amp)
    f['skew'] = skew(amp.flatten())
    f['kurtosis'] = kurtosis(amp.flatten())

    diff = np.diff(amp, axis=0)
    f['diff_mean'] = np.mean(np.abs(diff))
    f['diff_std'] = np.std(diff)
    f['spatial_var'] = np.mean(np.var(amp, axis=1))

    sig = np.mean(amp, axis=1)
    if len(sig) > 8:
        fft_mag = np.abs(np.fft.rfft(sig - np.mean(sig)))
        freqs = np.fft.rfftfreq(len(sig), 1/PACKET_RATE)
        f['spectral_energy'] = np.sum(fft_mag**2)
        mask = (freqs >= 0.1) & (freqs <= 2.0)
        f['motion_energy'] = np.sum(fft_mag[mask]**2) if np.any(mask) else 0
    else:
        f['spectral_energy'] = 0
        f['motion_energy'] = 0
    
    # Phase features
    phase_unwrapped = np.unwrap(phase, axis=1)
    f['phase_var'] = np.var(phase_unwrapped)
    f['phase_std'] = np.std(phase_unwrapped)
    
    phase_diff = np.diff(phase_unwrapped, axis=0)
    f['phase_diff_mean'] = np.mean(np.abs(phase_diff))
    f['phase_diff_std'] = np.std(phase_diff)
    f['phase_spatial_var'] = np.mean(np.var(phase_unwrapped, axis=1))
    f['phase_range'] = np.max(phase_unwrapped) - np.min(phase_unwrapped)
    
    return f


def create_dataset(data):
    win = int(WINDOW_SEC * PACKET_RATE)
    step = int(STEP_SEC * PACKET_RATE)
    features, labels = [], []

    for entries, label in data:
        if len(entries) < win:
            continue
        amp, phase = preprocess(entries)
        for start in range(0, len(amp) - win + 1, step):
            f = extract_features(amp[start:start+win], phase[start:start+win])
            features.append(f)
            labels.append(label)

    df = pd.DataFrame(features).fillna(0)
    return df.values, np.array(labels), list(df.columns)


if __name__ == '__main__':
    print("Loading data...")
    data = load_all_data()
    print(f"Loaded {len(data)} files")

    if len(data) == 0:
        print(f"\nNo data found! Make sure you have:")
        print(f"  {DATA_DIR}/no_person/*.csv")
        print(f"  {DATA_DIR}/person_present/*.csv")
        exit(1)

    X, y, feature_names = create_dataset(data)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)

    scores = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"CV Accuracy (on train): {scores.mean():.1%} (+/- {scores.std()*2:.1%})")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    label_names = [LABELS[i] for i in sorted(np.unique(y))]
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Retrain on full dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    # Feature importance
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1]
    print("\nTop features:")
    for i in range(min(5, len(feature_names))):
        print(f"  {feature_names[idx[i]]}: {importance[idx[i]]:.3f}")

    # Save model
    bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "labels": LABELS,
        "config": {
            "WINDOW_SEC": WINDOW_SEC,
            "STEP_SEC": STEP_SEC,
            "PACKET_RATE": PACKET_RATE,
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"\nModel saved to: {MODEL_PATH}")

    # Show plots
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=label_names).plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.show()
