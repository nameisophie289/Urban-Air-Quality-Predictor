import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import joblib

# Configuration
DATA_PATH = 'data/processed/pollution_weather_merged.csv'
MODEL_DIR = 'models'
REPORT_DIR = 'report/images'
METRICS_PATH = 'report/metrics.json'
SCALER_PATH = 'models/scaler.pkl'
TIME_STEPS = 24
EPOCHS = 100
BATCH_SIZE = 16

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    
    # Handle missing values: Forward fill then backward fill
    df = df.ffill().bfill()
    
    # Feature selection (using all available numerical columns generally, but let's be specific based on notebook likely usage)
    # The notebook output showed 'Sequence shape: (4907, 24, 24)', meaning 24 features were used.
    # Let's inspect columns from the csv first or assume all numeric columns are used.
    # Typically: pm25, pm10, no2, so2, co, o3, temperature, humidity, etc.
    # We'll use all numeric columns except target is pm25.
    
    # Separate features and target
    # Target is pm25
    target_col = 'pm25'
    
    # Ensure pm25 is in columns
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataset")
    
    # Use all columns as features
    feature_cols = df.columns.tolist()
    
    data_X = df[feature_cols].values
    data_y = df[target_col].values
    
    # Scaling
    print("Scaling data...")
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(data_X)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(data_y.reshape(-1, 1))
    
    # Save scaler for later used
    joblib.dump({'X': scaler_X, 'y': scaler_y}, SCALER_PATH)
    
    return X_scaled, y_scaled, scaler_y, feature_cols

def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        # Input sequences: past 'time_steps' hours
        X_seq.append(X[i : i + time_steps])
        # Target: value at 'time_steps' (next hour)
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape, units=[64, 32], dropout_rate=0.2):
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units[1], return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae', 'mse'])
    return model

def plot_history(history, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # MAE
    ax2.plot(history.history['mae'], label='Train')
    ax2.plot(history.history['val_mae'], label='Validation')
    ax2.set_title('Model MAE')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_predictions(y_true, y_pred, title, filename):
    plt.figure(figsize=(15, 6))
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('PM2.5')
    plt.legend()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def main():
    # 1. Load and Scale Data
    X_scaled, y_scaled, scaler_y, features = load_and_preprocess_data()
    
    # 2. Create Sequences
    print(f"Creating sequences with lookback = {TIME_STEPS}...")
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)
    print(f"Sequence shape: {X_seq.shape}")
    
    # 3. Train-Test Split (70-30 split based on notebook analysis)
    split_idx = int(0.7 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    # 4. Build Model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # 5. Train Model
    print("Starting training...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluation
    print("Evaluating model...")
    plot_history(history, 'training_history.png')
    
    # Predictions
    y_train_pred_scaled = model.predict(X_train)
    y_test_pred_scaled = model.predict(X_test)
    
    # Inverse transform
    y_train_true = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    y_test_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
    
    # Metrics
    metrics = {
        'train_mse': float(mean_squared_error(y_train_true, y_train_pred)),
        'train_mae': float(mean_absolute_error(y_train_true, y_train_pred)),
        'train_r2': float(r2_score(y_train_true, y_train_pred)),
        'test_mse': float(mean_squared_error(y_test_true, y_test_pred)),
        'test_mae': float(mean_absolute_error(y_test_true, y_test_pred)),
        'test_r2': float(r2_score(y_test_true, y_test_pred))
    }
    
    print("Metrics:", json.dumps(metrics, indent=4))
    
    # Save Metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # Save Model
    model_save_path = os.path.join(MODEL_DIR, 'lstm_pm25.h5')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Prediction Plots (Zoom in on a portion for clarity)
    plot_predictions(y_test_true[:200], y_test_pred[:200], 'PM2.5 Prediction (First 200 Test Samples)', 'prediction_plot_sample.png')
    plot_predictions(y_test_true, y_test_pred, 'PM2.5 Prediction (Full Test Set)', 'prediction_plot_full.png')

if __name__ == "__main__":
    main()
