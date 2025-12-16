# Local training script (no Colab dependencies)
# With comprehensive visualizations for academic paper

import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, learning_curve
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Output directory for figures
FIGURES_DIR = "./figures"

# LOCAL PATHS - adjust these
DATA_DIR = "./data"  # folder containing no_person/ and person_present/ subfolders
MODEL_PATH = "./csi_model_local.pkl"

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

LABELS = {0: "no_person", 1: "person_active", 2: "person_stationary"}
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


def plot_feature_importance(model, feature_names, top_n=15):
    importance = model.feature_importances_
    idx = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    
    bars = ax.barh(range(top_n), importance[idx][::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title('Random Forest Feature Importance for CSI-Based Presence Detection')
    
    # Add value labels
    for bar, val in zip(bars, importance[idx][::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png')
    return fig


def plot_roc_curve(model, X_test, y_test, label_names):
    fig, ax = plt.subplots(figsize=(8, 7))
    
    if len(label_names) == 2:
        # Binary classification
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='#2563eb', lw=2.5, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.fill_between(fpr, tpr, alpha=0.2, color='#2563eb')
    else:
        # Multi-class: one-vs-rest ROC
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(len(label_names)))
        y_score = model.predict_proba(X_test)
        
        colors = ['#2563eb', '#dc2626', '#16a34a']
        for i, (name, color) in enumerate(zip(label_names, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve for CSI Presence Detection')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/roc_curve.png')
    return fig


def plot_precision_recall_curve(model, X_test, y_test, label_names):
    fig, ax = plt.subplots(figsize=(8, 7))
    
    if len(label_names) == 2:
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        
        ax.plot(recall, precision, color='#7c3aed', lw=2.5,
                label=f'PR curve (AP = {ap:.3f})')
        ax.fill_between(recall, precision, alpha=0.2, color='#7c3aed')
    else:
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(len(label_names)))
        y_score = model.predict_proba(X_test)
        
        colors = ['#2563eb', '#dc2626', '#16a34a']
        for i, (name, color) in enumerate(zip(label_names, colors)):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
            ax.plot(recall, precision, color=color, lw=2,
                    label=f'{name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve for CSI Presence Detection')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/precision_recall_curve.png')
    return fig


def plot_feature_distributions(X, y, feature_names, label_names, top_n=8):
    importance_order = np.argsort(np.var(X, axis=0))[::-1][:top_n]
    
    n_cols = 4
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5*n_rows))
    axes = axes.flatten()
    
    colors = ['#22c55e', '#ef4444', '#3b82f6']  # Green for no_person, Red for person
    
    for i, feat_idx in enumerate(importance_order):
        ax = axes[i]
        data_by_class = [X[y == c, feat_idx] for c in range(len(label_names))]
        
        bp = ax.boxplot(data_by_class, patch_artist=True, labels=label_names)
        for patch, color in zip(bp['boxes'], colors[:len(label_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(feature_names[feat_idx], fontsize=11)
        ax.tick_params(axis='x', rotation=15)
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Feature Distributions by Class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_distributions.png')
    return fig


def plot_correlation_heatmap(X, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 8}, vmin=-1, vmax=1)
    
    ax.set_title('Feature Correlation Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png')
    return fig


def plot_tsne_visualization(X, y, label_names, perplexity=30):
    # Subsample if too many points
    max_samples = 2000
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sub)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#22c55e', '#ef4444', '#3b82f6']
    markers = ['o', 's', '^']
    
    for i, (name, color, marker) in enumerate(zip(label_names, colors, markers)):
        mask = y_sub == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
                   alpha=0.6, s=50, marker=marker, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of CSI Features')
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/tsne_visualization.png')
    return fig


def plot_pca_visualization(X, y, label_names):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax = axes[0]
    colors = ['#22c55e', '#ef4444', '#3b82f6']
    markers = ['o', 's', '^']
    
    for i, (name, color, marker) in enumerate(zip(label_names, colors, markers)):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                   alpha=0.6, s=50, marker=marker, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title('PCA Visualization of CSI Features')
    ax.legend(loc='best')
    
    # Explained variance plot
    ax2 = axes[1]
    pca_full = PCA().fit(X)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    
    ax2.bar(range(1, len(cumsum)+1), pca_full.explained_variance_ratio_, 
            alpha=0.7, color='#3b82f6', label='Individual')
    ax2.plot(range(1, len(cumsum)+1), cumsum, 'ro-', label='Cumulative')
    ax2.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95% threshold')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('PCA Explained Variance')
    ax2.legend()
    ax2.set_xlim(0.5, min(15, len(cumsum))+0.5)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/pca_visualization.png')
    return fig


def plot_learning_curve(model, X, y, cv=5):
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#2563eb')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='#dc2626')
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#2563eb', lw=2, label='Training Score')
    ax.plot(train_sizes_abs, val_mean, 's-', color='#dc2626', lw=2, label='Validation Score')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curve: Model Performance vs Training Data Size')
    ax.legend(loc='lower right')
    ax.set_ylim([0.5, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/learning_curve.png')
    return fig


def plot_cv_scores(scores, model_name="Random Forest"):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bp = ax.boxplot([scores], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#3b82f6')
    bp['boxes'][0].set_alpha(0.7)
    
    # Add individual points
    x = np.random.normal(1, 0.04, len(scores))
    ax.scatter(x, scores, alpha=0.7, color='#1e40af', s=100, zorder=3)
    
    ax.axhline(y=np.mean(scores), color='#dc2626', linestyle='--', 
               label=f'Mean: {np.mean(scores):.3f}', lw=2)
    
    ax.set_xticklabels([model_name])
    ax.set_ylabel('Accuracy')
    ax.set_title(f'5-Fold Cross-Validation Scores')
    ax.legend()
    ax.set_ylim([min(scores) - 0.05, 1.0])
    
    # Add stats text
    stats_text = f'Mean: {np.mean(scores):.3f}\nStd: {np.std(scores):.3f}'
    ax.text(1.3, np.mean(scores), stats_text, fontsize=11, va='center')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/cv_scores.png')
    return fig


def plot_confusion_matrix_enhanced(y_test, y_pred, label_names):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=label_names, yticklabels=label_names,
                annot_kws={'size': 14})
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('Confusion Matrix (Counts)')
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', ax=axes[1],
                xticklabels=label_names, yticklabels=label_names,
                annot_kws={'size': 14})
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/confusion_matrix.png')
    return fig


def plot_sample_csi_waveforms(data, n_samples=2):
    n_classes = len(LABELS)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(14, 4*n_classes))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    for label, dirname in LABELS.items():
        samples_plotted = 0
        for entries, entry_label in data:
            if entry_label != label or samples_plotted >= n_samples:
                continue
            
            amp, phase = preprocess(entries[:100])  # First 100 packets
            ax = axes[label, samples_plotted]
            
            # Plot subset of subcarriers
            for sc in range(0, min(amp.shape[1], 50), 5):
                ax.plot(amp[:, sc], alpha=0.7, linewidth=0.8)
            
            ax.set_title(f'{dirname} - Sample {samples_plotted+1}')
            ax.set_xlabel('Packet Index')
            ax.set_ylabel('Amplitude')
            samples_plotted += 1
    
    fig.suptitle('Sample CSI Amplitude Waveforms by Class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/sample_waveforms.png')
    return fig


def generate_classification_report_table(y_test, y_pred, label_names):
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    
    # Create DataFrame
    df = pd.DataFrame(report).transpose()
    df = df.round(3)
    
    # Save to CSV
    df.to_csv(f'{FIGURES_DIR}/classification_report.csv')
    
    # Create visual table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Format for display
    display_df = df.iloc[:-3]  # Exclude avg rows for main table
    
    table = ax.table(cellText=display_df.values,
                     colLabels=display_df.columns,
                     rowLabels=display_df.index,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#3b82f6')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    ax.set_title('Classification Report', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/classification_report.png')
    
    return df


def generate_dataset_summary_table(X, y, feature_names, label_names):
    summary = {
        'Metric': ['Total Samples', 'Number of Features', 'Number of Classes'],
        'Value': [len(X), len(feature_names), len(label_names)]
    }
    
    # Add class distribution
    for i, name in enumerate(label_names):
        count = np.sum(y == i)
        pct = count / len(y) * 100
        summary['Metric'].append(f'Class: {name}')
        summary['Value'].append(f'{count} ({pct:.1f}%)')
    
    df = pd.DataFrame(summary)
    df.to_csv(f'{FIGURES_DIR}/dataset_summary.csv', index=False)
    
    # Feature statistics
    feature_stats = pd.DataFrame(X, columns=feature_names).describe().round(4)
    feature_stats.to_csv(f'{FIGURES_DIR}/feature_statistics.csv')
    
    return df, feature_stats


if __name__ == '__main__':
    print("Loading data...")
    data = load_all_data()
    print(f"Loaded {len(data)} files")

    if len(data) == 0:
        print(f"No data found in {DATA_DIR}/")
        exit(1)

    X, y, feature_names = create_dataset(data)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    label_names = [LABELS[i] for i in sorted(np.unique(y))]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"CV Accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Retrain on full dataset
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)
    model.fit(X_scaled_final, y)

    # Save model
    bundle = {
        "model": model,
        "scaler": scaler_final,
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
    print(f"Model saved: {MODEL_PATH}")

    # Generate visualizations
    print(f"\nGenerating figures to {FIGURES_DIR}/")
    
    generate_dataset_summary_table(X, y, feature_names, label_names)
    
    try:
        plot_sample_csi_waveforms(data, n_samples=2)
    except Exception as e:
        print(f"Skipped waveforms: {e}")
    
    plot_feature_distributions(X_scaled, y, feature_names, label_names, top_n=8)
    plot_correlation_heatmap(X_scaled, feature_names)
    plot_pca_visualization(X_scaled, y, label_names)
    plot_tsne_visualization(X_scaled, y, label_names)
    
    model_for_viz = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
    model_for_viz.fit(X_train_scaled, y_train)
    
    plot_feature_importance(model_for_viz, feature_names)
    plot_learning_curve(model_for_viz, X_scaled, y, cv=5)
    plot_cv_scores(cv_scores)
    
    y_pred_viz = model_for_viz.predict(X_test_scaled)
    plot_confusion_matrix_enhanced(y_test, y_pred_viz, label_names)
    plot_roc_curve(model_for_viz, X_test_scaled, y_test, label_names)
    plot_precision_recall_curve(model_for_viz, X_test_scaled, y_test, label_names)
    generate_classification_report_table(y_test, y_pred_viz, label_names)

    print("\nDone. Displaying plots...")
    plt.show()
