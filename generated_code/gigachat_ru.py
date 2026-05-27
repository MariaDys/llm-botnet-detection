import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def load_data(data_dir, device_id):
    """Загрузка данных для указанного устройства"""
    benign_path = f"{data_dir}/{device_id}.benign.csv"
    if not os.path.exists(benign_path):
        print(f"Ошибка: файл {benign_path} не найден.")
        return None, None
    
    normal_df = pd.read_csv(benign_path)
    attack_dfs = []
    
    # Поиск файлов атак
    for filename in os.listdir(data_dir):
        if filename.startswith(device_id) and ".benign." not in filename:
            path = os.path.join(data_dir, filename)
            df = pd.read_csv(path)
            attack_dfs.append(df)
            
    if not attack_dfs:
        print(f"Ошибка: файлы атак для устройства {device_id} не найдены.")
        return None, None
        
    attacks_df = pd.concat(attack_dfs, ignore_index=True)
    
    # Добавление меток класса
    normal_df['label'] = 0
    attacks_df['label'] = 1
    
    combined_df = pd.concat([normal_df, attacks_df])
    
    # Проверка количества признаков
    if len(combined_df.columns) != 116:  # 115 признаков + label
        print("Ошибка: неверное количество признаков в данных.")
        return None, None
        
    return normal_df.drop('label', axis=1), attacks_df.drop('label', axis=1)

def preprocess_data(normal_train, normal_val, test_data):
    """Предварительная обработка данных"""
    # Замена Inf на NaN
    normal_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    normal_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Заполнение пропущенных значений средним значением столбца
    normal_train.fillna(normal_train.mean(), inplace=True)
    normal_val.fillna(normal_val.mean(), inplace=True)
    test_data.fillna(test_data.mean(), inplace=True)
    
    # Масштабирование данных
    scaler = MinMaxScaler()
    scaled_normal_train = scaler.fit_transform(normal_train)
    scaled_normal_val = scaler.transform(normal_val)
    scaled_test_data = scaler.transform(test_data)
    
    return scaled_normal_train, scaled_normal_val, scaled_test_data

def build_autoencoder(input_dim):
    """Создание глубокой автоэнкодерной модели"""
    input_layer = Input(shape=(input_dim,))
    
    # Encoder layers
    x = Dense(int(input_dim * 1.0))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(int(input_dim * 0.75))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(int(input_dim * 0.5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(int(input_dim * 0.33))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    encoded = Dense(int(input_dim * 0.25))(x)
    
    # Decoder layers
    x = Dense(int(input_dim * 0.33))(encoded)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(int(input_dim * 0.5))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(int(input_dim * 0.75))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    decoded = Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def main():
    parser = argparse.ArgumentParser(description="Обнаружение ботнет-атак в IoT-трафике")
    parser.add_argument("--data_dir", required=True, help="Путь к каталогу с CSV-файлами")
    parser.add_argument("--device_id", required=True, help="Идентификатор устройства")
    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох обучения")
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    parser.add_argument("--seed", type=int, default=42, help="Значение для инициализации генераторов случайных чисел")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Установка случайного числа
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Загрузка данных
    normal_data, attack_data = load_data(args.data_dir, args.device_id)
    if normal_data is None or attack_data is None:
        sys.exit(1)
    
    # Разделение данных
    X_normal_train, X_normal_test = train_test_split(
        normal_data.values, test_size=0.33, random_state=args.seed
    )
    X_attack_test = attack_data.values
    
    X_test = np.concatenate((X_normal_test, X_attack_test))
    y_test = np.array([0]*len(X_normal_test) + [1]*len(X_attack_test))
    
    X_normal_train, X_normal_val = train_test_split(
        X_normal_train, test_size=0.2, random_state=args.seed
    )
    
    # Предобработка данных
    X_train_scaled, X_val_scaled, X_test_scaled = preprocess_data(
        pd.DataFrame(X_normal_train),
        pd.DataFrame(X_normal_val),
        pd.DataFrame(X_test)
    )
    
    # Создание и обучение модели
    model = build_autoencoder(X_train_scaled.shape[1])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]
    
    history = model.fit(
        X_train_scaled, X_train_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=(X_val_scaled, X_val_scaled),
        verbose=1,
        callbacks=callbacks
    )
    
    # Определение порога аномальности
    val_reconstructions = model.predict(X_val_scaled)
    mse = np.mean(np.power(X_val_scaled - val_reconstructions, 2), axis=1)
    threshold = np.mean(mse) + np.std(mse)
    
    # Тестирование модели
    predictions = model.predict(X_test_scaled)
    errors = np.mean(np.power(X_test_scaled - predictions, 2), axis=1)
    y_pred = np.where(errors >= threshold, 1, 0)
    
    print(classification_report(y_test, y_pred))
    print(f"Точность: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    
    # Визуализация
    plt.figure(figsize=(10,5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend(loc='upper right')
    plt.title("Динамика потерь")
    plt.show()
    
    plt.hist(errors[y_test==0], bins=50, alpha=0.5, label="Нормальный трафик")
    plt.hist(errors[y_test==1], bins=50, alpha=0.5, label="Атаки")
    plt.axvline(threshold, color='k', linestyle='dashed', linewidth=1, label=f"Порог ({threshold:.2f})")
    plt.legend(loc='upper right')
    plt.title("Распределение ошибок реконструкции")
    plt.xlabel("Среднеквадратичная ошибка")
    plt.ylabel("Частота")
    plt.show()
    
    end_time = time.time()
    print(f"Время выполнения: {(end_time-start_time)/60:.2f} минут")

if __name__ == "__main__":
    main()