import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

def load_data(data_dir, device_id):
    """
    Loads and processes data for specified device.
    
    :param data_dir: Path to directory containing CSV files
    :param device_id: Device identifier prefix
    :return: Tuple of (X_train, X_val, X_test, y_test)
    """
    # Load benign data
    benign_path = f"{data_dir}/{device_id}.benign.csv"
    if not os.path.exists(benign_path):
        raise FileNotFoundError(f"Benign file {benign_path} not found")
    benign_df = pd.read_csv(benign_path)
    benign_df["label"] = 0

    # Load attack data
    attack_paths = (
        glob(f"{data_dir}/{device_id}.gafgyt*.csv") +
        glob(f"{data_dir}/{device_id}.mirai*.csv")
    )
    if not attack_paths:
        raise ValueError("No attack files found")
    attack_dfs = []
    for path in attack_paths:
        df = pd.read_csv(path)
        df["label"] = 1
        attack_dfs.append(df)
    attack_df = pd.concat(attack_dfs, ignore_index=True)

    # Check feature count
    if len(benign_df.columns) != 116 or len(attack_df.columns) != 116:
        raise ValueError("Incorrect number of columns")

    return benign_df.drop(columns=["label"]).values.astype(np.float32), \
           attack_df.drop(columns=["label"]).values.astype(np.float32)

def preprocess_data(X_benign, X_attack):
    """
    Preprocesses data by handling infinities, splitting datasets, scaling.
    
    :param X_benign: Benign traffic data
    :param X_attack: Attack traffic data
    :return: Processed splits and scaler
    """
    # Handle infinities
    X_benign[np.isinf(X_benign)] = np.nan
    X_attack[np.isinf(X_attack)] = np.nan

    # Impute missing values
    benign_mean = np.nanmean(X_benign, axis=0)
    X_benign = np.where(np.isnan(X_benign), benign_mean, X_benign)
    X_attack = np.where(np.isnan(X_attack), benign_mean, X_attack)

    # Split benign data
    X_benign_train_full, X_benign_test = train_test_split(
        X_benign, test_size=1/3, random_state=args.seed
    )
    X_benign_train, X_benign_val = train_test_split(
        X_benign_train_full, test_size=0.2, random_state=args.seed
    )

    # Create test set
    X_test = np.concatenate([X_benign_test, X_attack])
    y_test = np.array([0]*len(X_benign_test) + [1]*len(X_attack))

    # Scale data
    scaler = MinMaxScaler()
    scaler.fit(X_benign_train)
    X_benign_train_scaled = scaler.transform(X_benign_train)
    X_benign_val_scaled = scaler.transform(X_benign_val)
    X_test_scaled = scaler.transform(X_test)

    return X_benign_train_scaled, X_benign_val_scaled, X_test_scaled, y_test, scaler

def build_autoencoder(input_dim):
    """
    Builds a deep autoencoder model.
    
    :param input_dim: Dimensionality of input data
    :return: Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs

    # Encoder layers
    x = tf.keras.layers.Dense(int(input_dim * 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dense(int(input_dim * 0.75))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dense(int(input_dim * 0.5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dense(int(input_dim * 0.33))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    encoded = tf.keras.layers.Dense(int(input_dim * 0.25))(x)
    
    # Decoder layers
    x = tf.keras.layers.Dense(int(input_dim * 0.33))(encoded)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dense(int(input_dim * 0.5))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    x = tf.keras.layers.Dense(int(input_dim * 0.75))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    
    decoded = tf.keras.layers.Dense(input_dim)(x)

    autoencoder = tf.keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def compute_threshold(autoencoder, X_val):
    """
    Computes anomaly detection threshold based on validation data.
    
    :param autoencoder: Trained autoencoder model
    :param X_val: Validation data
    :return: Threshold value
    """
    val_pred = autoencoder.predict(X_val)
    val_mses = np.mean((val_pred - X_val)**2, axis=1)
    mean_mse = np.mean(val_mses)
    std_mse = np.std(val_mses)
    return mean_mse + std_mse

def evaluate_model(autoencoder, X_test, y_test, threshold):
    """
    Evaluates model performance on test data.
    
    :param autoencoder: Trained autoencoder model
    :param X_test: Test data
    :param y_test: True labels
    :param threshold: Anomaly detection threshold
    """
    test_pred = autoencoder.predict(X_test)
    test_mses = np.mean((test_pred - X_test)**2, axis=1)
    predictions = (test_mses > threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print("\nAccuracy Score:", accuracy_score(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion Matrix:\n", cm)

def visualize_results(history, test_mses, y_test, threshold):
    """
    Plots training history and reconstruction errors.
    
    :param history: Training history object
    :param test_mses: Test reconstruction errors
    :param y_test: True labels
    :param threshold: Detection threshold
    """
    # Loss plot
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    # Histogram plot
    benign_errors = test_mses[y_test == 0]
    attack_errors = test_mses[y_test == 1]
    
    plt.figure(figsize=(10,5))
    sns.histplot(benign_errors, kde=True, color="blue", label="Benign")
    sns.histplot(attack_errors, kde=True, color="orange", label="Attack")
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Mean Squared Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    start_time = datetime.now()
    
    parser = argparse.ArgumentParser(description="Anomaly-Based Botnet Detection Script")
    parser.add_argument("--data_dir", required=True, help="Path to data directory")
    parser.add_argument("--device_id", required=True, help="Device identifier prefix")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    try:
        # Step 1: Load data
        X_benign, X_attack = load_data(args.data_dir, args.device_id)
        
        # Step 2: Preprocess data
        X_train, X_val, X_test, y_test, scaler = preprocess_data(X_benign, X_attack)
        
        # Step 3: Build model
        autoencoder = build_autoencoder(X_train.shape[1])
        
        # Step 4: Train model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5)
        ]
        history = autoencoder.fit(
            X_train, X_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_val, X_val),
            shuffle=True,
            verbose=1,
            callbacks=callbacks
        )
        
        # Step 5: Compute threshold
        threshold = compute_threshold(autoencoder, X_val)
        print(f"\nComputed threshold: {threshold}")
        
        # Step 6: Evaluate model
        evaluate_model(autoencoder, X_test, y_test, threshold)
        
        # Step 7: Visualize results
        test_pred = autoencoder.predict(X_test)
        test_mses = np.mean((test_pred - X_test)**2, axis=1)
        visualize_results(history, test_mses, y_test, threshold)
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print(f"\nTotal execution time: {(datetime.now() - start_time)}")