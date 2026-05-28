"""
Botnet Detection: ручные модели

  1. Deep Autoencoder
  2. Isolation Forest
  3. One-Class SVM

Сравнение:
  - LLM автоэнкодер vs ручной автоэнкодер
  - Deep learning (автоэнкодер) vs классический ML (IsoForest, OC-SVM)
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
import argparse


# запуск через чтение аргументов командной строки: python3 baseline/botnet_detection_manual.py --data_dir ДАТАСЕТ --device_id НОМЕР_УСТРОЙСТВА
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="dataset")
parser.add_argument("--device_id", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
DATA_DIR = args.data_dir
DEVICE_ID = args.device_id
RANDOM_SEED = args.seed


# загрузка benign и attack данных из датасета для конкретного устройства
def load_device_data(data_dir, device_id):
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


# разделение данных на train (только benign) и test (benign + attacks)
def prepare_splits(benign, attacks):
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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)

    return X_train, X_val, X_test, y_test


# ============== метод 1: ISOLATION FOREST ==============

#ml без учителя, разработан для обнаружения аномалий (выбросов) в данных, тренируется только на нормальных (не attack) данных
def run_isolation_forest(X_train, X_test, y_test):
  
    print("\n" + "=" * 50)
    print("METHOD 2: ISOLATION FOREST")
    print("=" * 50)
  
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

    # IsolationForest возвращает 1 для normal, -1 для anomaly
    raw_pred = clf.predict(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)  # convert to 0=benign, 1=attack

    print(classification_report(y_test.astype(int), y_pred,
                                target_names=["Benign", "Attack"]))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return accuracy_score(y_test, y_pred)


# ============== метод 2: ONE-CLASS SVM ==============

# One-Class SVM строит границы вокруг нормального поведения, детектирует таким ообразом отклонение. Работает медленно на больших датасетах.
def run_ocsvm(X_train, X_test, y_test):
  
    print("\n" + "=" * 50)
    print("METHOD 3: ONE-CLASS SVM")
    print("=" * 50)

    # O(n^2)–O(n^3), поэтому используем подвыборку
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


# ============== Сравнение ==============

#таблица со сравнением
def compare_results(results: dict):
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

    # Load data
    benign, attacks = load_device_data(DATA_DIR, DEVICE_ID)
    X_train, X_val, X_test, y_test = prepare_splits(benign, attacks)

    results = {}

    # 1 -- Isolation Forest
    if_acc = run_isolation_forest(X_train, X_test, y_test)
    results["Isolation Forest"] = if_acc

    # 2 -- One-Class SVM
    ocsvm_acc = run_ocsvm(X_train, X_test, y_test)
    results["One-Class SVM (subsampled)"] = ocsvm_acc

    # сравнение
    compare_results(results)

    elapsed = time.time() - start
    print(f"\nTotal execution time: {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()
