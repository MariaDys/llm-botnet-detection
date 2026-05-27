#!/usr/bin/env python3
"""
==============================================================================
IoT Botnet Detection via Deep Autoencoder — N-BaIoT Dataset
==============================================================================

Detects botnet attacks (Mirai & BASHLITE/Gafgyt) in IoT network traffic using
a deep autoencoder trained exclusively on benign traffic. Anomalies are flagged
when reconstruction error (MSE) exceeds a learned threshold.

Based on: Meidan et al. (2018), "N-BaIoT — Network-Based Detection of IoT
Botnet Attacks Using Deep Autoencoders", IEEE Pervasive Computing.

Dataset: UCI ML Repository ID 442 — 9 IoT devices, 115 flow-statistics features.

Usage:
    python botnet_autoencoder.py --data_dir ./data --device_id "Danmini_Doorbell"
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF info/warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks


# ──────────────────────────────────────────────────────────────────────────────
# 1. CLI Arguments
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="IoT Botnet Detection via Deep Autoencoder (N-BaIoT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing N-BaIoT CSV files.",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        required=True,
        help="Device identifier used as filename prefix (e.g. 'Danmini_Doorbell').",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory where plots and results are saved.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Data Loading
# ──────────────────────────────────────────────────────────────────────────────
EXPECTED_FEATURES = 115


def _load_csv(path: str, label: int) -> pd.DataFrame:
    """Load a single CSV, attach an integer label column, and validate width."""
    df = pd.read_csv(path, low_memory=False)
    if df.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{path}: expected {EXPECTED_FEATURES} feature columns, got {df.shape[1]}"
        )
    df["label"] = label
    return df


def load_device_data(data_dir: str, device_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load benign and attack CSVs for *device_id*.

    Returns
    -------
    benign_df : DataFrame  — benign traffic (label=0), no 'label' column
    attack_df : DataFrame  — all attack traffic (label=1), no 'label' column
    """
    data_dir = Path(data_dir)

    # --- Benign ---
    benign_path = data_dir / f"{device_id}.benign.csv"
    if not benign_path.exists():
        sys.exit(f"[ERROR] Benign file not found: {benign_path}")
    print(f"  ✓ Loading benign traffic : {benign_path.name}")
    benign_df = _load_csv(str(benign_path), label=0)

    # --- BASHLITE / Gafgyt attacks ---
    gafgyt_pattern = str(data_dir / f"{device_id}.gafgyt.*.csv")
    gafgyt_files = sorted(glob.glob(gafgyt_pattern))

    # --- Mirai attacks ---
    mirai_pattern = str(data_dir / f"{device_id}.mirai.*.csv")
    mirai_files = sorted(glob.glob(mirai_pattern))

    if not gafgyt_files and not mirai_files:
        sys.exit(
            f"[ERROR] No attack files found for device '{device_id}' in {data_dir}.\n"
            f"        Searched patterns:\n"
            f"          {gafgyt_pattern}\n"
            f"          {mirai_pattern}"
        )

    attack_frames: list[pd.DataFrame] = []
    for f in gafgyt_files:
        print(f"  ✓ Loading Gafgyt attack  : {Path(f).name}")
        attack_frames.append(_load_csv(f, label=1))
    for f in mirai_files:
        print(f"  ✓ Loading Mirai attack   : {Path(f).name}")
        attack_frames.append(_load_csv(f, label=1))

    attack_df = pd.concat(attack_frames, ignore_index=True)

    # Drop the helper label column; we track labels externally
    benign_df = benign_df.drop(columns=["label"])
    attack_df = attack_df.drop(columns=["label"])

    print(
        f"\n  Benign samples : {len(benign_df):>10,}\n"
        f"  Attack samples : {len(attack_df):>10,}\n"
        f"  Features       : {benign_df.shape[1]:>10}"
    )
    return benign_df, attack_df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(
    benign_df: pd.DataFrame,
    attack_df: pd.DataFrame,
    seed: int,
) -> dict:
    """
    Clean, split, and scale the data.

    Split strategy
    --------------
    benign  ─┬─ 2/3 train pool ─┬─ 80 % → X_train  (autoencoder fitting)
              │                  └─ 20 % → X_val    (threshold calibration)
              └─ 1/3 ──────────────────── → part of X_test (label = 0)

    attack  ───────────────────────────── → part of X_test (label = 1)
    """
    print("[3] Preprocessing …")

    # --- 3a. Replace Inf with NaN, then fill remaining NaN with 0 ---
    for df in (benign_df, attack_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    benign_nan = benign_df.isna().sum().sum()
    attack_nan = attack_df.isna().sum().sum()
    benign_df.fillna(0.0, inplace=True)
    attack_df.fillna(0.0, inplace=True)
    print(f"  Replaced Inf→NaN→0: benign {benign_nan:,} | attack {attack_nan:,} cells")

    # --- 3b. Split benign into train-pool (2/3) and test-benign (1/3) ---
    benign_train_pool, benign_test = train_test_split(
        benign_df, test_size=1 / 3, random_state=seed
    )

    # --- 3c. Split train-pool into train (80 %) and validation (20 %) ---
    X_train, X_val = train_test_split(
        benign_train_pool, test_size=0.2, random_state=seed
    )

    # --- 3d. Build test set: leftover benign + all attacks ---
    X_test = pd.concat([benign_test, attack_df], ignore_index=True)
    y_test = np.concatenate(
        [np.zeros(len(benign_test)), np.ones(len(attack_df))]
    )

    print(
        f"  Train      (benign only) : {len(X_train):>10,}\n"
        f"  Validation (benign only) : {len(X_val):>10,}\n"
        f"  Test       (mixed)       : {len(X_test):>10,}  "
        f"(benign {len(benign_test):,} + attack {len(attack_df):,})"
    )

    # --- 3e. MinMaxScaler fitted on train only ---
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("  MinMaxScaler fitted on training data and applied to all splits.")

    return {
        "X_train": X_train.astype(np.float32),
        "X_val": X_val.astype(np.float32),
        "X_test": X_test.astype(np.float32),
        "y_test": y_test.astype(np.int32),
        "n_benign_test": len(benign_test),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Autoencoder Architecture
# ──────────────────────────────────────────────────────────────────────────────
def build_autoencoder(input_dim: int = EXPECTED_FEATURES) -> keras.Model:
    """
    Deep autoencoder with symmetric encoder/decoder.

    Encoder widths (fraction of input_dim):
        100 % → 75 % → 50 % → 33 % → **25 % (bottleneck)**
    Decoder mirrors the encoder.
    """
    enc_units = [
        input_dim,                          # 115  (100 %)
        int(round(input_dim * 0.75)),       #  86  ( 75 %)
        int(round(input_dim * 0.50)),       #  58  ( 50 %)
        int(round(input_dim * 0.33)),       #  38  ( 33 %)
    ]
    bottleneck = int(round(input_dim * 0.25))  # 29  ( 25 %)

    inp = keras.Input(shape=(input_dim,), name="input")
    x = inp

    # ── Encoder ──
    for i, units in enumerate(enc_units):
        x = layers.Dense(units, name=f"enc_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"enc_bn_{i}")(x)
        x = layers.LeakyReLU(name=f"enc_lrelu_{i}")(x)

    # ── Bottleneck ──
    x = layers.Dense(bottleneck, name="bottleneck")(x)
    x = layers.BatchNormalization(name="bottleneck_bn")(x)
    x = layers.LeakyReLU(name="bottleneck_lrelu")(x)

    # ── Decoder (symmetric) ──
    dec_units = list(reversed(enc_units))
    for i, units in enumerate(dec_units):
        x = layers.Dense(units, name=f"dec_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"dec_bn_{i}")(x)
        x = layers.LeakyReLU(name=f"dec_lrelu_{i}")(x)

    # ── Output (linear activation) ──
    output = layers.Dense(input_dim, activation="linear", name="output")(x)

    model = keras.Model(inputs=inp, outputs=output, name="DeepAutoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 5. Training
# ──────────────────────────────────────────────────────────────────────────────
def train_autoencoder(
    model: keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int,
    batch_size: int,
) -> keras.callbacks.History:
    """Train the autoencoder on benign traffic only (input == target)."""
    print("[5] Training autoencoder …\n")

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        X_train,               # Target == input for autoencoders
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1,
    )
    return history


# ──────────────────────────────────────────────────────────────────────────────
# 6. Anomaly Threshold
# ──────────────────────────────────────────────────────────────────────────────
def compute_threshold(model: keras.Model, X_val: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Compute anomaly threshold as mean + 1·std of per-sample MSE on validation
    (benign) data.
    """
    reconstructed = model.predict(X_val, verbose=0)
    mse_val = np.mean((X_val - reconstructed) ** 2, axis=1)
    threshold = float(np.mean(mse_val) + np.std(mse_val))
    print(
        f"  Validation MSE — mean: {np.mean(mse_val):.6f}  "
        f"std: {np.std(mse_val):.6f}\n"
        f"  ► Anomaly threshold (mean + std): {threshold:.6f}"
    )
    return threshold, mse_val


# ──────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify test samples and print metrics."""
    print("[7] Evaluating on test set …\n")
    reconstructed = model.predict(X_test, verbose=0)
    mse_test = np.mean((X_test - reconstructed) ** 2, axis=1)

    y_pred = (mse_test > threshold).astype(int)

    print("  Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Attack"],
            digits=4,
        )
    )
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0, 0]:>9,}  FP={cm[0, 1]:>9,}")
    print(f"    FN={cm[1, 0]:>9,}  TP={cm[1, 1]:>9,}")
    return mse_test, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# 8. Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def plot_results(
    history: keras.callbacks.History,
    mse_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    output_dir: str,
) -> None:
    """Save training curves and reconstruction-error histogram."""
    os.makedirs(output_dir, exist_ok=True)

    # ── 8a. Training & validation loss ──
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train loss")
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
    print(f"  ✓ Saved training curves     → {loss_path}")

    # ── 8b. Reconstruction error histogram ──
    mse_benign = mse_test[y_test == 0]
    mse_attack = mse_test[y_test == 1]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Use log scale for MSE values to better separate distributions
    eps = 1e-12
    log_benign = np.log10(mse_benign + eps)
    log_attack = np.log10(mse_attack + eps)
    log_thresh = np.log10(threshold + eps)

    ax.hist(log_benign, bins=120, alpha=0.6, label="Benign", color="#2196F3", density=True)
    ax.hist(log_attack, bins=120, alpha=0.6, label="Attack", color="#F44336", density=True)
    ax.axvline(log_thresh, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold ({threshold:.4e})")
    ax.set_xlabel("log₁₀(Reconstruction MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution — Benign vs Attack")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    hist_path = os.path.join(output_dir, "reconstruction_error_hist.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved error histogram     → {hist_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    wall_start = time.perf_counter()
    args = parse_args()

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("=" * 70)
    print(" IoT Botnet Detection — Deep Autoencoder (N-BaIoT)")
    print("=" * 70)
    print(f"  Device       : {args.device_id}")
    print(f"  Data dir     : {args.data_dir}")
    print(f"  Epochs (max) : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Random seed  : {args.seed}")
    print(f"  Output dir   : {args.output_dir}")
    print("=" * 70)

    # Step 2 — Load data
    print("\n[2] Loading data …")
    benign_df, attack_df = load_device_data(args.data_dir, args.device_id)

    # Step 3 — Preprocess
    print()
    data = preprocess(benign_df, attack_df, seed=args.seed)

    # Step 4 — Build model
    print("\n[4] Building autoencoder …")
    model = build_autoencoder(EXPECTED_FEATURES)
    model.summary(print_fn=lambda s: print(f"  {s}"))

    # Step 5 — Train
    print()
    history = train_autoencoder(
        model, data["X_train"], data["X_val"], args.epochs, args.batch_size
    )

    # Step 6 — Threshold
    print("\n[6] Computing anomaly threshold …")
    threshold, mse_val = compute_threshold(model, data["X_val"])

    # Step 7 — Evaluate
    print()
    mse_test, y_pred = evaluate(model, data["X_test"], data["y_test"], threshold)

    # Step 8 — Plots
    print("\n[8] Generating visualisations …")
    plot_results(history, mse_test, data["y_test"], threshold, args.output_dir)

    # Step 9 — Elapsed time
    elapsed = time.perf_counter() - wall_start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n{'=' * 70}")
    print(f" Done. Total wall time: {int(minutes)}m {seconds:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()