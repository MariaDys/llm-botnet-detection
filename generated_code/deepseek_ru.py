#!/usr/bin/env python3
"""
Botnet Attack Detection in IoT Traffic using a Deep Autoencoder (Anomaly Detection).

Reference:
    Meidan et al. (2018), "N-BaIoT: Network-based Detection of IoT Botnet Attacks
    Using Deep Autoencoders", IEEE Pervasive Computing.

Dataset: N-BaIoT from UCI Machine Learning Repository (ID 442).
Each CSV file contains 115 numerical traffic statistics.

The script:
  1. Loads benign and attack traffic for a given device.
  2. Preprocesses data (handles Inf/NaN, splits, scales).
  3. Builds a deep symmetrical autoencoder.
  4. Trains only on benign data.
  5. Computes an anomaly threshold from validation reconstruction error.
  6. Evaluates on a test set containing held-out benign and all attacks.
  7. Outputs metrics and visualisations.
"""

import argparse
import glob
import os
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    LeakyReLU,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Anomaly-based botnet detection using a deep autoencoder."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the N-BaIoT CSV files.",
    )
    parser.add_argument(
        "--device_id",
        type=str,
        required=True,
        help="Device identifier (prefix of the CSV files), e.g. 'Danmini_Doorbell'.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs. (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size. (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. (default: 42)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=False,
        help="Save the generated plots to disk instead of showing them.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save plots if --save_plots is set. (default: current dir)",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_device_data(data_dir, device_id):
    """
    Load benign and attack CSV files for a single device.

    Returns:
        benign_df (DataFrame): all benign records with label 0.
        attack_df (DataFrame): all attack records with label 1.
    """
    # Load benign
    benign_path = os.path.join(data_dir, f"{device_id}.benign.csv")
    if not os.path.isfile(benign_path):
        raise FileNotFoundError(f"Benign file not found: {benign_path}")
    benign_df = pd.read_csv(benign_path)
    if benign_df.shape[1] != 115:
        # Sometimes an index column is prepended – drop it if extra columns exist
        if benign_df.shape[1] == 116:
            benign_df = benign_df.iloc[:, 1:]  # drop first column
        else:
            raise ValueError(
                f"Expected 115 features, got {benign_df.shape[1]} in {benign_path}"
            )
    benign_df["label"] = 0
    print(f"[INFO] Loaded benign data: {benign_df.shape[0]} samples.")

    # Collect attack files
    attack_patterns = [
        os.path.join(data_dir, f"{device_id}.gafgyt.*.csv"),
        os.path.join(data_dir, f"{device_id}.mirai.*.csv"),
    ]
    attack_files = []
    for pattern in attack_patterns:
        attack_files.extend(glob.glob(pattern))

    if not attack_files:
        print("[WARNING] No attack files found. Test set will contain only benign traffic.")

    attack_dfs = []
    for filepath in attack_files:
        df = pd.read_csv(filepath)
        if df.shape[1] != 115:
            if df.shape[1] == 116:
                df = df.iloc[:, 1:]
            else:
                raise ValueError(
                    f"Expected 115 features, got {df.shape[1]} in {filepath}"
                )
        df["label"] = 1
        attack_dfs.append(df)
    if attack_dfs:
        attack_df = pd.concat(attack_dfs, ignore_index=True)
        print(f"[INFO] Loaded attack data: {attack_df.shape[0]} samples from {len(attack_files)} files.")
    else:
        attack_df = pd.DataFrame(columns=benign_df.columns)  # empty with same columns
    return benign_df, attack_df


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def preprocess_data(benign_df, attack_df, seed):
    """
    Split, impute missing values, and scale the data.

    Returns:
        X_train, X_val, X_test, y_test and the fitted scaler/imputer.
    """
    # Separate features and labels
    X_benign = benign_df.drop("label", axis=1).values.astype(np.float64)
    X_attack = attack_df.drop("label", axis=1).values.astype(np.float64)
    y_attack = attack_df["label"].values.astype(int)

    # Replace Inf/-Inf with NaN
    X_benign = np.where(np.isinf(X_benign), np.nan, X_benign)
    X_attack = np.where(np.isinf(X_attack), np.nan, X_attack)

    # Split benign into train (2/3) and held-out test normal (1/3)
    X_benign_trainval, X_benign_test = train_test_split(
        X_benign, test_size=1 / 3, random_state=seed, shuffle=True
    )

    # Further split trainval into train (80%) and validation (20%)
    X_train, X_val = train_test_split(
        X_benign_trainval, test_size=0.2, random_state=seed, shuffle=True
    )

    # Test set = held-out benign + all attacks
    X_test = np.vstack([X_benign_test, X_attack]) if X_attack.size > 0 else X_benign_test
    y_test = np.concatenate([np.zeros(X_benign_test.shape[0]), y_attack]) if X_attack.size > 0 else np.zeros(X_benign_test.shape[0])

    print(f"[INFO] Train normal: {X_train.shape[0]}, Val normal: {X_val.shape[0]}, "
          f"Test normal: {X_benign_test.shape[0]}, Test attack: {X_attack.shape[0]}")

    # Impute NaN with median computed on training data only
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scale to [0, 1] using MinMaxScaler fitted on training data only
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_test, scaler, imputer


# ----------------------------------------------------------------------
# Autoencoder model
# ----------------------------------------------------------------------
def build_autoencoder(input_dim):
    """
    Build a deep symmetrical autoencoder.

    Encoder: 100% -> 75% -> 50% -> 33% -> 25% (bottleneck)
    Each encoder block: Dense -> BatchNormalization -> LeakyReLU.
    Decoder: symmetric reverse order.
    """
    # Encoder
    input_layer = Input(shape=(input_dim,), name="input")
    x = input_layer

    # Encoding stages (percentages of input_dim)
    encoder_units = [
        int(input_dim * 1.0),
        int(input_dim * 0.75),
        int(input_dim * 0.50),
        int(input_dim * 0.33),
    ]
    bottleneck_units = int(input_dim * 0.25)

    for i, units in enumerate(encoder_units):
        x = Dense(units, name=f"enc_dense_{i}")(x)
        x = BatchNormalization(name=f"enc_bn_{i}")(x)
        x = LeakyReLU(name=f"enc_leaky_{i}")(x)

    # Bottleneck
    x = Dense(bottleneck_units, name="bottleneck_dense")(x)
    x = BatchNormalization(name="bottleneck_bn")(x)
    x = LeakyReLU(name="bottleneck_leaky")(x)

    # Decoder (symmetric)
    decoder_units = encoder_units[::-1]
    for i, units in enumerate(decoder_units):
        x = Dense(units, name=f"dec_dense_{i}")(x)
        x = BatchNormalization(name=f"dec_bn_{i}")(x)
        x = LeakyReLU(name=f"dec_leaky_{i}")(x)

    # Output layer – linear activation, reconstruct original values
    output_layer = Dense(input_dim, name="output")(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="autoencoder")
    return autoencoder


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_autoencoder(model, X_train, X_val, epochs, batch_size, seed):
    """Compile and train the autoencoder."""
    model.compile(optimizer=Adam(), loss="mse")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        X_train,  # autoencoder target = input
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate_model(model, X_train, X_val, X_test, y_test):
    """
    Compute anomaly threshold on validation data and evaluate on test set.
    Returns predictions, metrics, and error arrays.
    """
    # Reconstruction errors
    train_pred = model.predict(X_train, verbose=0)
    train_mse = np.mean(np.square(X_train - train_pred), axis=1)

    val_pred = model.predict(X_val, verbose=0)
    val_mse = np.mean(np.square(X_val - val_pred), axis=1)

    test_pred = model.predict(X_test, verbose=0)
    test_mse = np.mean(np.square(X_test - test_pred), axis=1)

    # Threshold = mean + std of validation MSE
    threshold = np.mean(val_mse) + np.std(val_mse)
    print(f"[INFO] Anomaly threshold (mean+std): {threshold:.6f}")

    # Binary predictions
    y_pred = (test_mse > threshold).astype(int)

    # Print metrics
    print("\n" + "=" * 50)
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=["benign", "attack"]))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    return y_pred, test_mse, threshold, train_mse, val_mse, history


# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
def plot_results(history, test_mse, y_test, threshold, save_plots, output_dir):
    """Generate and save/show loss curves and error histograms."""
    # Loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    if save_plots:
        path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Loss curve saved to {path}")
    else:
        plt.show()

    # Histogram of reconstruction errors
    benign_errors = test_mse[y_test == 0]
    attack_errors = test_mse[y_test == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(benign_errors, bins=50, alpha=0.6, label="Benign", color="green")
    if len(attack_errors) > 0:
        plt.hist(attack_errors, bins=50, alpha=0.6, label="Attack", color="red")
    plt.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.4f})")
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    if save_plots:
        path = os.path.join(output_dir, "reconstruction_error_hist.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Error histogram saved to {path}")
    else:
        plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 1. Load data
    print("[STEP 1] Loading data ...")
    benign_df, attack_df = load_device_data(args.data_dir, args.device_id)

    # 2. Preprocess
    print("[STEP 2] Preprocessing ...")
    X_train, X_val, X_test, y_test, scaler, imputer = preprocess_data(
        benign_df, attack_df, args.seed
    )

    # 3. Build autoencoder
    print("[STEP 3] Building autoencoder ...")
    input_dim = X_train.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.summary()

    # 4. Train
    print("[STEP 4] Training autoencoder (benign only) ...")
    history = train_autoencoder(
        autoencoder, X_train, X_val, args.epochs, args.batch_size, args.seed
    )

    # 5. Evaluate
    print("[STEP 5] Evaluating ...")
    y_pred, test_mse, threshold, train_mse, val_mse, history = evaluate_model(
        autoencoder, X_train, X_val, X_test, y_test
    )

    # 6. Plot
    print("[STEP 6] Generating plots ...")
    plot_results(history, test_mse, y_test, threshold, args.save_plots, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total execution time: {timedelta(seconds=elapsed)}")


if __name__ == "__main__":
    main()