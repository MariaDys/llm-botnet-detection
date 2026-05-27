#!/usr/bin/env python3
"""
Anomaly-Based Botnet Detection in IoT Network Traffic
======================================================
Deep autoencoder trained exclusively on benign IoT traffic (N-BaIoT methodology,
Meidan et al. 2018).  Reconstruction MSE serves as the anomaly score; samples
whose error exceeds a learned threshold are classified as attacks.

Dataset : N-BaIoT  –  UCI ML Repository ID 442
Features: 115 numerical network-flow statistics per record
Botnets : Mirai  &  BASHLITE / Gafgyt

Usage
-----
    python botnet_detector.py --data_dir ./data --device_id Danmini_Doorbell

Run  ``python botnet_detector.py -h``  for the full list of options.
"""

# ──────────────────────────────────────────────────────────────
# 0.  Imports
# ──────────────────────────────────────────────────────────────
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend – safe for headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info/warning noise
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Input,
    LeakyReLU,
)

EXPECTED_FEATURES = 115

# ──────────────────────────────────────────────────────────────
# 1.  Argument parsing
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Anomaly-based IoT botnet detector (deep autoencoder, N-BaIoT)."
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the folder containing the N-BaIoT CSV files.",
    )
    p.add_argument(
        "--device_id",
        type=str,
        required=True,
        help="Device identifier prefix used in CSV filenames "
        "(e.g. 'Danmini_Doorbell').",
    )
    p.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100).")
    p.add_argument("--batch_size", type=int, default=256, help="Training batch size (default: 256).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for plots and outputs (default: results/).",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# 2.  Data loading
# ──────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    """Load a single CSV and validate its column count."""
    df = pd.read_csv(path)
    if df.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{path}: expected {EXPECTED_FEATURES} features, found {df.shape[1]}."
        )
    return df


def load_device_data(data_dir: str, device_id: str):
    """
    Return (X_benign, X_attack, attack_labels) for the requested device.

    attack_labels is a list of human-readable strings such as
    'gafgyt.combo' or 'mirai.syn' – one per attack DataFrame row.
    """
    # --- Benign ---
    benign_path = os.path.join(data_dir, f"{device_id}.benign.csv")
    if not os.path.isfile(benign_path):
        sys.exit(f"[ERROR] Benign file not found: {benign_path}")
    print(f"  Loading benign traffic  : {benign_path}")
    df_benign = _load_csv(benign_path)
    print(f"    → {len(df_benign):,} samples, {df_benign.shape[1]} features")

    # --- Attack files (gafgyt + mirai) ---
    attack_pattern = os.path.join(data_dir, f"{device_id}.gafgyt.*.csv")
    mirai_pattern = os.path.join(data_dir, f"{device_id}.mirai.*.csv")
    attack_files = sorted(glob.glob(attack_pattern) + glob.glob(mirai_pattern))

    if not attack_files:
        sys.exit(
            f"[ERROR] No attack CSVs found for device '{device_id}' in {data_dir}.\n"
            f"  Searched patterns:\n    {attack_pattern}\n    {mirai_pattern}"
        )

    attack_frames = []
    attack_labels = []
    for fpath in attack_files:
        fname = os.path.basename(fpath)
        # Extract label: everything between device_id. and .csv
        label = fname.replace(f"{device_id}.", "").replace(".csv", "")
        print(f"  Loading attack traffic  : {fname:<50s}", end="")
        df = _load_csv(fpath)
        print(f"  → {len(df):>8,} samples  [{label}]")
        attack_frames.append(df)
        attack_labels.extend([label] * len(df))

    X_benign = df_benign.values.astype(np.float64)
    X_attack = np.vstack([df.values.astype(np.float64) for df in attack_frames])
    return X_benign, X_attack, np.array(attack_labels)


# ──────────────────────────────────────────────────────────────
# 3.  Preprocessing
# ──────────────────────────────────────────────────────────────

def preprocess(X_benign, X_attack, seed: int):
    """
    Clean, split, and scale the data.

    Returns
    -------
    X_train, X_val, X_test, y_test, scaler
    """
    # Replace Inf → NaN, then forward-fill remaining NaN with column median
    for arr in (X_benign, X_attack):
        arr[~np.isfinite(arr)] = np.nan

    col_medians = np.nanmedian(X_benign, axis=0)
    for arr in (X_benign, X_attack):
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_medians, inds[1])

    # Split benign: 2/3 for AE training pool, 1/3 held out for testing
    benign_train_pool, benign_test = train_test_split(
        X_benign, test_size=1 / 3, random_state=seed
    )
    # From the 2/3 pool: 80% train, 20% validation
    X_train, X_val = train_test_split(
        benign_train_pool, test_size=0.2, random_state=seed
    )

    # Test set = held-out benign + all attack samples
    X_test = np.vstack([benign_test, X_attack])
    y_test = np.concatenate(
        [np.zeros(len(benign_test)), np.ones(len(X_attack))]
    )

    print(f"\n  Train (benign)    : {X_train.shape[0]:>9,} samples")
    print(f"  Validation (benign): {X_val.shape[0]:>9,} samples")
    print(f"  Test  (benign)    : {benign_test.shape[0]:>9,} samples")
    print(f"  Test  (attack)    : {X_attack.shape[0]:>9,} samples")
    print(f"  Test  (total)     : {X_test.shape[0]:>9,} samples")

    # Fit scaler on training split ONLY
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_test, scaler


# ──────────────────────────────────────────────────────────────
# 4.  Autoencoder architecture
# ──────────────────────────────────────────────────────────────

def build_autoencoder(input_dim: int = EXPECTED_FEATURES) -> Model:
    """
    Deep symmetric autoencoder.

    Encoder widths : 115 → 115 → 86 → 57 → 38 → 29  (bottleneck)
    Decoder widths : 29  → 38  → 57 → 86 → 115 → 115
    """
    widths = [
        int(input_dim * r)
        for r in (1.00, 0.75, 0.50, 0.33)
    ]
    bottleneck = int(input_dim * 0.25)

    # --- Encoder ---
    inp = Input(shape=(input_dim,), name="input")
    x = inp
    for i, w in enumerate(widths):
        x = Dense(w, name=f"enc_dense_{i}")(x)
        x = BatchNormalization(name=f"enc_bn_{i}")(x)
        x = LeakyReLU(name=f"enc_act_{i}")(x)

    # Bottleneck
    x = Dense(bottleneck, name="bottleneck")(x)
    x = BatchNormalization(name="bottleneck_bn")(x)
    x = LeakyReLU(name="bottleneck_act")(x)

    # --- Decoder (symmetric) ---
    for i, w in enumerate(reversed(widths)):
        x = Dense(w, name=f"dec_dense_{i}")(x)
        x = BatchNormalization(name=f"dec_bn_{i}")(x)
        x = LeakyReLU(name=f"dec_act_{i}")(x)

    # Linear output (reconstruction)
    output = Dense(input_dim, activation="linear", name="output")(x)

    model = Model(inp, output, name="DeepAutoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# ──────────────────────────────────────────────────────────────
# 5.  Training
# ──────────────────────────────────────────────────────────────

def train_autoencoder(model, X_train, X_val, epochs, batch_size):
    """Train autoencoder; return the Keras History object."""
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        X_train,
        X_train,  # target == input (reconstruction objective)
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────────────────────
# 6.  Anomaly threshold
# ──────────────────────────────────────────────────────────────

def compute_threshold(model, X_val):
    """
    Threshold = mean(MSE_val) + std(MSE_val)
    where MSE_val is the per-sample reconstruction error on benign validation data.
    """
    recon = model.predict(X_val, verbose=0)
    mse_val = np.mean((X_val - recon) ** 2, axis=1)
    threshold = mse_val.mean() + mse_val.std()
    print(f"\n  Validation MSE  –  mean: {mse_val.mean():.6f}  std: {mse_val.std():.6f}")
    print(f"  Anomaly threshold       : {threshold:.6f}")
    return threshold, mse_val


# ──────────────────────────────────────────────────────────────
# 7.  Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold):
    """Compute per-sample MSE on test set, classify, and print metrics."""
    recon = model.predict(X_test, verbose=0)
    mse_test = np.mean((X_test - recon) ** 2, axis=1)

    y_pred = (mse_test > threshold).astype(int)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION  REPORT")
    print("=" * 60)
    print(
        classification_report(
            y_test, y_pred, target_names=["Benign", "Attack"], digits=4
        )
    )
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):\n{cm}\n")
    return mse_test, y_pred


# ──────────────────────────────────────────────────────────────
# 8.  Visualisation
# ──────────────────────────────────────────────────────────────

def plot_results(history, mse_test, y_test, threshold, output_dir):
    """Save training curves and reconstruction-error histogram."""
    os.makedirs(output_dir, exist_ok=True)

    # --- 8a. Training & validation loss ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, "training_loss.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"  Saved training-loss plot → {loss_path}")

    # --- 8b. Reconstruction-error histogram ---
    benign_mse = mse_test[y_test == 0]
    attack_mse = mse_test[y_test == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(benign_mse, bins=200, alpha=0.65, label="Benign", color="#2196F3", density=True)
    ax.hist(attack_mse, bins=200, alpha=0.65, label="Attack", color="#F44336", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.5f}")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Zoom x-axis to the interesting region (cap at 99.5th percentile of attack)
    upper = np.percentile(attack_mse, 99.5)
    ax.set_xlim(0, max(upper, threshold * 3))

    fig.tight_layout()
    hist_path = os.path.join(output_dir, "error_histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Saved histogram plot    → {hist_path}")


# ──────────────────────────────────────────────────────────────
# 9.  Main entry point
# ──────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("\n" + "=" * 60)
    print("  IoT Botnet Detector  –  Deep Autoencoder (N-BaIoT)")
    print("=" * 60)

    # ── Load ──
    print(f"\n[1/6] Loading data for device '{args.device_id}' …")
    X_benign, X_attack, attack_labels = load_device_data(args.data_dir, args.device_id)

    # ── Preprocess ──
    print("\n[2/6] Preprocessing …")
    X_train, X_val, X_test, y_test, scaler = preprocess(
        X_benign, X_attack, seed=args.seed
    )

    # ── Build model ──
    print("\n[3/6] Building autoencoder …")
    model = build_autoencoder(EXPECTED_FEATURES)
    model.summary()

    # ── Train ──
    print("\n[4/6] Training autoencoder …")
    history = train_autoencoder(model, X_train, X_val, args.epochs, args.batch_size)

    # ── Threshold ──
    print("\n[5/6] Computing anomaly threshold …")
    threshold, _ = compute_threshold(model, X_val)

    # ── Evaluate ──
    print("\n[6/6] Evaluating on test set …")
    mse_test, y_pred = evaluate(model, X_test, y_test, threshold)

    # ── Plots ──
    print("\n  Generating plots …")
    plot_results(history, mse_test, y_test, threshold, args.output_dir)

    elapsed = time.time() - t_start
    print(f"\n  Total execution time: {elapsed:.1f} s  ({elapsed / 60:.2f} min)")
    print("  Done.\n")


if __name__ == "__main__":
    main()