#!/usr/bin/env python3
"""
N-BaIoT Botnet Detection using Deep Autoencoder
=================================================

Implements the approach of Meidan et al. (2018) for IoT network traffic
anomaly detection. A deep autoencoder is trained solely on benign traffic,
and the reconstruction error (MSE) is used as an anomaly score. Samples
whose reconstruction MSE exceeds a threshold (mean + std of validation
benign MSE) are flagged as attacks.

Dataset: N-BaIoT (UCI Machine Learning Repository, ID 442)
         https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT

Usage:
    python nba_botnet_detector.py --data_dir /path/to/dataset --device_id 1
"""

import os
import sys
import time
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Suppress unnecessary warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Constants
EXPECTED_FEATURES = 115


def parse_args():
    parser = argparse.ArgumentParser(
        description="N-BaIoT anomaly detection via deep autoencoder"
    )
    parser.add_argument('--data_dir', required=True,
                        help='Path to the directory containing CSV files')
    parser.add_argument('--device_id', required=True, type=str,
                        help='Device identifier (e.g., "1") used in file names')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_device_data(device_id, data_dir):
    """
    Load benign and attack CSV files for a given device.

    Returns:
        X_benign (DataFrame): benign samples
        X_attack (DataFrame): all attack samples (Gafgyt + Mirai)
    """
    print(f"\n[1] Loading data for device '{device_id}' from {data_dir} ...")

    # Benign file
    benign_pattern = os.path.join(data_dir, f'{device_id}.benign.csv')
    benign_files = glob.glob(benign_pattern)
    if not benign_files:
        raise FileNotFoundError(
            f"Benign file not found: {benign_pattern}"
        )
    benign_file = benign_files[0]
    print(f"    Benign: {os.path.basename(benign_file)}")

    # Attack files: Gafgyt and Mirai
    gafgyt_pattern = os.path.join(data_dir, f'{device_id}.gafgyt.*.csv')
    mirai_pattern = os.path.join(data_dir, f'{device_id}.mirai.*.csv')
    gafgyt_files = sorted(glob.glob(gafgyt_pattern))
    mirai_files = sorted(glob.glob(mirai_pattern))
    attack_files = gafgyt_files + mirai_files

    if not attack_files:
        print("    WARNING: No attack files found. Test set will contain only "
              "benign samples.", file=sys.stderr)

    # Load benign data
    benign_df = pd.read_csv(benign_file)
    print(f"    Benign shape: {benign_df.shape}")

    # Load and combine all attack data
    attack_dfs = []
    for f in attack_files:
        df = pd.read_csv(f)
        attack_dfs.append(df)
        print(f"    Attack: {os.path.basename(f)}  shape: {df.shape}")

    if attack_dfs:
        attack_df = pd.concat(attack_dfs, ignore_index=True)
        print(f"    Combined attack shape: {attack_df.shape}")
    else:
        attack_df = pd.DataFrame(columns=benign_df.columns)  # empty with same cols

    # Validate feature count
    if benign_df.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, got {benign_df.shape[1]}"
        )

    X_benign = benign_df.values.astype(np.float32)
    X_attack = attack_df.values.astype(np.float32)

    return X_benign, X_attack


def preprocess(X_benign, X_attack, seed):
    """
    Replace infinite values, drop NaN rows, split and scale data.

    Returns:
        X_train, X_val, X_test (numpy arrays)
        y_test (0 for benign, 1 for attack)
        scaler (fitted MinMaxScaler)
    """
    print("\n[2] Preprocessing...")

    # Replace Inf/-Inf with NaN and drop rows containing NaN
    def clean_array(arr, name):
        arr = np.where(np.isinf(arr), np.nan, arr)
        nan_mask = np.isnan(arr).any(axis=1)
        n_dropped = nan_mask.sum()
        if n_dropped > 0:
            print(f"    Dropped {n_dropped} rows with NaN from {name}")
        return arr[~nan_mask]

    X_benign = clean_array(X_benign, "benign")
    X_attack = clean_array(X_attack, "attack")

    # Split benign: 2/3 for training+validation, 1/3 held out for test
    train_val_benign, holdout_benign = train_test_split(
        X_benign, test_size=1/3, random_state=seed, shuffle=True
    )
    print(f"    Benign train_val: {train_val_benign.shape[0]}, "
          f"holdout: {holdout_benign.shape[0]}")

    # Further split train_val into training (80%) and validation (20%)
    X_train, X_val = train_test_split(
        train_val_benign, test_size=0.2, random_state=seed, shuffle=True
    )
    print(f"    Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

    # Build test set: held-out benign + all attacks
    if X_attack.size == 0:
        X_test = holdout_benign
        y_test = np.zeros(len(holdout_benign), dtype=int)
        print("    Test set: benign only (no attacks loaded)")
    else:
        X_test = np.vstack([holdout_benign, X_attack])
        y_test = np.hstack([
            np.zeros(len(holdout_benign), dtype=int),
            np.ones(len(X_attack), dtype=int)
        ])
        print(f"    Test set: {X_test.shape[0]} samples "
              f"({len(holdout_benign)} benign, {len(X_attack)} attacks)")

    # Scale to [0, 1] using training data only
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("    Scaling complete.")
    return X_train, X_val, X_test, y_test, scaler


def build_autoencoder(input_dim=115):
    """
    Build symmetric deep autoencoder.

    Architecture:
        Encoder: Dense(115) -> BN -> LeakyReLU -> Dense(86) -> BN -> LeakyReLU
                 -> Dense(57) -> BN -> LeakyReLU -> Dense(38) -> BN -> LeakyReLU
        Bottleneck: Dense(28) -> BN -> LeakyReLU
        Decoder: Dense(38) -> BN -> LeakyReLU -> Dense(57) -> BN -> LeakyReLU
                 -> Dense(86) -> BN -> LeakyReLU -> Dense(115, linear)
    """
    print("\n[3] Building autoencoder ...")

    # Layer sizes as percentages of input dim
    enc_dims = [
        input_dim,
        int(np.ceil(input_dim * 0.75)),
        int(np.ceil(input_dim * 0.50)),
        int(np.ceil(input_dim * 0.33))
    ]
    bottleneck_dim = int(np.ceil(input_dim * 0.25))

    model = models.Sequential(name='N-BaIoT_Autoencoder')

    # Encoder
    for i, units in enumerate(enc_dims):
        if i == 0:
            model.add(layers.InputLayer(input_shape=(input_dim,)))
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.1))

    # Bottleneck
    model.add(layers.Dense(bottleneck_dim))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))

    # Decoder (symmetric, excluding input layer)
    dec_dims = enc_dims[::-1]
    for units in dec_dims:
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.1))

    # Output layer (linear activation)
    model.add(layers.Dense(input_dim, activation='linear'))

    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(learning_rate=0.001)
    )
    model.summary()
    return model


def train_autoencoder(model, X_train, X_val, args):
    """Train the autoencoder only on benign data."""
    print("\n[4] Training autoencoder (only on benign traffic) ...")

    cb = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, X_train,                # input = target
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, X_val),
        callbacks=cb,
        verbose=1
    )
    return history


def compute_threshold(model, X_val):
    """Compute anomaly threshold as mean(MSE) + std(MSE) on validation benign."""
    print("\n[5] Computing anomaly threshold ...")

    X_pred = model.predict(X_val, verbose=0)
    mse = np.mean(np.square(X_val - X_pred), axis=1)
    threshold = mse.mean() + mse.std()
    print(f"    Validation MSE mean: {mse.mean():.6f}, std: {mse.std():.6f}")
    print(f"    Anomaly threshold: {threshold:.6f}")
    return threshold, mse


def evaluate(model, X_test, y_test, threshold):
    """
    Evaluate on test set: compute MSE, classify using threshold,
    and print metrics.
    """
    print("\n[6] Evaluating on test set ...")

    X_pred = model.predict(X_test, verbose=0)
    mse_test = np.mean(np.square(X_test - X_pred), axis=1)

    y_pred = (mse_test > threshold).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return mse_test, y_pred


def plot_training_history(history):
    """Plot training and validation loss."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    plt.close()
    print("    Saved training_loss.png")


def plot_reconstruction_error(mse_test, y_test, threshold):
    """Plot histogram of reconstruction error for benign and attack samples."""
    plt.figure(figsize=(8, 5))
    benign_mse = mse_test[y_test == 0]
    attack_mse = mse_test[y_test == 1]

    bins = np.linspace(0, max(mse_test.max(), threshold*1.5), 50)
    plt.hist(benign_mse, bins=bins, alpha=0.6, label='Benign', color='green',
             edgecolor='black')
    plt.hist(attack_mse, bins=bins, alpha=0.6, label='Attack', color='red',
             edgecolor='black')

    plt.axvline(threshold, color='blue', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.6f})')
    plt.xlabel('Reconstruction MSE')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig('reconstruction_error.png', dpi=150)
    plt.close()
    print("    Saved reconstruction_error.png")


def main():
    start_time = time.time()
    args = parse_args()
    set_seed(args.seed)

    # Load data
    X_benign, X_attack = load_device_data(args.device_id, args.data_dir)

    # Preprocess and split
    X_train, X_val, X_test, y_test, _ = preprocess(X_benign, X_attack, args.seed)

    # Build autoencoder
    model = build_autoencoder(input_dim=EXPECTED_FEATURES)

    # Train
    history = train_autoencoder(model, X_train, X_val, args)

    # Threshold
    threshold, val_mse = compute_threshold(model, X_val)

    # Evaluate
    mse_test, y_pred = evaluate(model, X_test, y_test, threshold)

    # Visualize
    print("\n[7] Generating plots ...")
    plot_training_history(history)
    plot_reconstruction_error(mse_test, y_test, threshold)

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()