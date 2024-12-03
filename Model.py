import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

# Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyperparameters and Configuration
config = {
    "epochs": 50,
    "batch_size": 32,
    "patience": 10,
    "sequence_length": 10,
    "data_path": "LiveData/cleaned_BTC_1d.csv",
    "model_save_path": "Model/lstm_model.keras",
    "results_save_path":"result.json"
}

# Create necessary folders
os.makedirs("Model", exist_ok=True)

# Load the dataset
data = pd.read_csv(config["data_path"])

# Extract features and target
features = ["Open", "High", "Low", "Close", "Volume", "Open Interest (USD)"]
target = "Close"

# Scale the data
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_features = feature_scaler.fit_transform(data[features])
scaled_target = target_scaler.fit_transform(data[[target]])

# Create sequences for LSTM
def create_sequences(features, target, sequence_length):
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, scaled_target, config["sequence_length"])

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
early_stopping = EarlyStopping(monitor='val_loss', patience=config["patience"], restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=config["epochs"],
    batch_size=config["batch_size"],
    callbacks=[early_stopping],
    verbose=1
)

# Make predictions
predicted_scaled = model.predict(X_test)

# Inverse scale predictions and actual values
predicted = target_scaler.inverse_transform(predicted_scaled)
actual = target_scaler.inverse_transform(y_test)

# Create a comparison array
comparison = np.concatenate([actual, predicted], axis=1)
comparison_df = pd.DataFrame(comparison, columns=["Actual", "Predicted"])

# Print the first few rows of the comparison for verification
print("First 10 rows of Actual vs Predicted values after inverse scaling:")
print(comparison_df.head(10))

# Calculate evaluation metrics
metrics = {
    "MSE": mean_squared_error(actual, predicted),
    "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    "MAE": mean_absolute_error(actual, predicted),
    "MAPE": np.mean(np.abs((actual - predicted) / actual)) * 100,
    "R2": r2_score(actual, predicted)
}

# Print evaluation metrics
print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Save the model
save_model(model, config["model_save_path"])
print(f"Model saved to {config['model_save_path']}.")

# Save results to result.json
results = {
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss'],
    "metrics": metrics,
    "predictions": predicted.flatten().tolist(),
    "actual": actual.flatten().tolist(),
    "comparison": comparison.tolist()
}

with open(config["results_save_path"], "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {config['results_save_path']}.")
