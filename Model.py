import sys
print(f"Python executable: {sys.executable}")
print("Python Path:", sys.path)
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

# Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Create necessary folders
os.makedirs("Model", exist_ok=True)

# Load the dataset
file_path = "LiveData/cleaned_BTC_1d.csv"
data = pd.read_csv(file_path)

# Extract features and target
features = ["Open", "High", "Low", "Close", "Volume", "Open Interest (USD)"]
target = "Close"

# Scale the data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_features = feature_scaler.fit_transform(data[features])
scaled_target = target_scaler.fit_transform(data[[target]])

# Create sequences for LSTM
def create_sequences(features, target, sequence_length=10):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(scaled_features, scaled_target, sequence_length)

# Split into training and testing datasets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
predicted_scaled = model.predict(X_test)

# Inverse scale predictions and actual values
predicted = target_scaler.inverse_transform(predicted_scaled)
actual = target_scaler.inverse_transform(y_test)

# Save loss and validation loss
results = {
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss'],
    "predictions": predicted.flatten().tolist(),
    "actual": actual.flatten().tolist()
}

# Save results to result.json
with open("result.json", "w") as f:
    json.dump(results, f)

# Save the model in the 'model' folder as 'lstm_model.h5'
model_path = "model/lstm_model.keras"
save_model(model, model_path)
print(f"LSTM model saved to {model_path}")
