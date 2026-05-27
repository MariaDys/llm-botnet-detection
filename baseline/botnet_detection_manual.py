"""
Botnet Detection on N-BaIoT — MANUALLY CODED BASELINE
======================================================
This script is written "by hand" (without LLM prompting) to serve as
a comparison baseline for the LLM-generated autoencoder solution.

It implements THREE different approaches:
  1. Deep Autoencoder (similar architecture, slight differences)
  2. Isolation Forest (classical anomaly detection)
  3. One-Class SVM (classical anomaly detection)

This allows comparing:
  - LLM-generated autoencoder vs manually-coded autoencoder
  - Deep learning (autoencoder) vs classical ML (IsoForest, OC-SVM)
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # NOTE: StandardScaler, not MinMax
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ========================== CONFIGURATION ==========================
DATA_DIR = "dataset"
DEVICE_ID = 1
RANDOM_SEED = 42
# ===================================================================


def load_device_data(data_dir, device_id):
    """Load benign and attack data for a device."""
    print(f"Loading data for device {device_id}...")

    benign_path = os.path.join(data_dir, f"{device_id}.benign.csv")
    if not os.path.exists(benign_path):
        benign_files = glob.glob(os.path.join(data_dir, f"{device_id}*.benign*.csv"))
        benign_path = benign_files[0] if benign_files else None
    if benign_path is None:
        raise FileNotFoundError(f"No benign data for device {device_id}")

    benign = pd.read_csv(benign_path)

    attack_files = (
        glob.glob(os.path.join(data_dir, f"{device_id}.gafgyt.*.csv")) +
        glob.glob(os.path.join(data_dir, f"{device_id}.mirai.*.csv"))
    )
    attacks = pd.concat([pd.read_csv(f) for f in sorted(attack_files)], ignore_index=True)

    print(f"  Benign: {len(benign):,}  |  Attack: {len(attacks):,}")
    return benign, attacks


def prepare_splits(benign, attacks):
    """Split data into train (benign only) / test (benign + attacks)."""
    benign_vals = benign.values
    np.random.seed(RANDOM_SEED)
    perm = np.random.permutation(len(benign_vals))
    cut = int(len(benign_vals) * 2 / 3)

    X_train_all = benign_vals[perm[:cut]]
    X_test_benign = benign_vals[perm[cut:]]
    X_test_attack = attacks.values

    X_test = np.vstack([X_test_benign, X_test_attack])
    y_test = np.concatenate([np.zeros(len(X_test_benign)), np.ones(len(X_test_attack))])

    X_train, X_val = train_test_split(X_train_all, test_size=0.2, random_state=RANDOM_SEED)

    # Use StandardScaler (difference from LLM version which uses MinMaxScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)

    return X_train, X_val, X_test, y_test


# ===================== METHOD 1: AUTOENCODER =====================

def build_manual_autoencoder(n_features):
    """
    Manually coded autoencoder — differs from the LLM version:
      - Uses Dropout instead of BatchNormalization
      - Uses 'relu' instead of LeakyReLU
      - Different layer sizes (powers of 2)
      - Uses StandardScaler instead of MinMaxScaler
    """
    inp = Input(shape=(n_features,))

    # Encoder with Dropout (different from LLM's BatchNorm + LeakyReLU)
    e = Dense(128, activation='relu')(inp)
    e = Dropout(0.2)(e)
    e = Dense(64, activation='relu')(e)
    e = Dropout(0.2)(e)
    e = Dense(32, activation='relu')(e)

    # Bottleneck
    bottleneck = Dense(16, activation='relu')(e)

    # Decoder
    d = Dense(32, activation='relu')(bottleneck)
    d = Dropout(0.2)(d)
    d = Dense(64, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(128, activation='relu')(d)

    output = Dense(n_features, activation='linear')(d)

    model = Model(inp, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def run_autoencoder(X_train, X_val, X_test, y_test):
    """Train autoencoder and evaluate."""
    print("\n" + "=" * 50)
    print("METHOD 1: MANUAL AUTOENCODER (Dropout + ReLU)")
    print("=" * 50)

    model = build_manual_autoencoder(X_train.shape[1])
    model.summary()

    history = model.fit(
        X_train, X_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_val, X_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
        verbose=1
    )

    # Threshold from validation
    val_pred = model.predict(X_val, verbose=0)
    val_mse = np.mean((X_val - val_pred) ** 2, axis=1)
    threshold = val_mse.mean() + val_mse.std()
    print(f"  Threshold: {threshold:.6f}")

    # Test
    test_pred = model.predict(X_test, verbose=0)
    test_mse = np.mean((X_test - test_pred) ** 2, axis=1)
    y_pred = (test_mse > threshold).astype(int)

    print(classification_report(y_test.astype(int), y_pred,
                                target_names=["Benign", "Attack"]))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return history, accuracy_score(y_test, y_pred)


# ============== METHOD 2: ISOLATION FOREST ==============

def run_isolation_forest(X_train, X_test, y_test):
    """
    Isolation Forest — classical unsupervised anomaly detector.
    Trained only on benign (normal) data.
    """
    print("\n" + "=" * 50)
    print("METHOD 2: ISOLATION FOREST")
    print("=" * 50)

    # Subsample training data if too large (IsoForest is slow on >100k samples)
    if len(X_train) > 50000:
        idx = np.random.choice(len(X_train), 50000, replace=False)
        X_fit = X_train[idx]
    else:
        X_fit = X_train

    clf = IsolationForest(
        n_estimators=200,
        contamination=0.01,  # assume ~1% anomalies in training (should be 0 ideally)
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1
    )
    clf.fit(X_fit)

    # Predict: IsolationForest returns 1 for normal, -1 for anomaly
    raw_pred = clf.predict(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)  # convert to 0=benign, 1=attack

    print(classification_report(y_test.astype(int), y_pred,
                                target_names=["Benign", "Attack"]))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return accuracy_score(y_test, y_pred)


# ============== METHOD 3: ONE-CLASS SVM ==============

def run_ocsvm(X_train, X_test, y_test):
    """
    One-Class SVM — trained only on benign data.
    NOTE: Very slow on large datasets; we subsample heavily.
    """
    print("\n" + "=" * 50)
    print("METHOD 3: ONE-CLASS SVM")
    print("=" * 50)

    # OC-SVM is O(n^2)–O(n^3), so we must subsample
    max_train = 10000
    max_test = 20000

    if len(X_train) > max_train:
        idx = np.random.choice(len(X_train), max_train, replace=False)
        X_fit = X_train[idx]
    else:
        X_fit = X_train

    if len(X_test) > max_test:
        idx = np.random.choice(len(X_test), max_test, replace=False)
        X_test_sub = X_test[idx]
        y_test_sub = y_test[idx]
    else:
        X_test_sub = X_test
        y_test_sub = y_test

    print(f"  Training on {len(X_fit):,} samples, testing on {len(X_test_sub):,} samples")

    clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    clf.fit(X_fit)

    raw_pred = clf.predict(X_test_sub)
    y_pred = np.where(raw_pred == -1, 1, 0)

    print(classification_report(y_test_sub.astype(int), y_pred,
                                target_names=["Benign", "Attack"]))
    print(f"Accuracy: {accuracy_score(y_test_sub, y_pred):.4f}")

    return accuracy_score(y_test_sub, y_pred)


# ============== COMPARISON SUMMARY ==============

def compare_results(results: dict):
    """Print a comparison table of all methods."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<35} {'Accuracy':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for method, acc in results.items():
        print(f"  {method:<35} {acc:>10.4f}")
    print("=" * 60)


def main():
    start = time.time()

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Load data
    benign, attacks = load_device_data(DATA_DIR, DEVICE_ID)
    X_train, X_val, X_test, y_test = prepare_splits(benign, attacks)

    results = {}

    # Method 1: Autoencoder
    _, ae_acc = run_autoencoder(X_train, X_val, X_test, y_test)
    results["Manual Autoencoder (Dropout+ReLU)"] = ae_acc

    # Method 2: Isolation Forest
    if_acc = run_isolation_forest(X_train, X_test, y_test)
    results["Isolation Forest"] = if_acc

    # Method 3: One-Class SVM (on subsampled data)
    ocsvm_acc = run_ocsvm(X_train, X_test, y_test)
    results["One-Class SVM (subsampled)"] = ocsvm_acc

    # Final comparison
    compare_results(results)

    elapsed = time.time() - start
    print(f"\nTotal execution time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()