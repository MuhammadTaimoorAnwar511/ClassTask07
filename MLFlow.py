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
import mlflow

# Suppress TensorFlow Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define configuration in a dictionary for central control
config = {
    "epochs": 200,
    "batch_size": 64,
    "patience": 10,
    "sequence_length": 4,
    "model_save_path": "MLFLOW/Model",
    "data_path": "LiveData/cleaned_BTC_1d.csv",
    "features": ["Open", "High", "Low", "Close", "Volume", "Open Interest (USD)"],
    "target": "Close"
}

# Create necessary folders
os.makedirs(config["model_save_path"], exist_ok=True)

# Load the dataset
data = pd.read_csv(config["data_path"])

# Extract features and target
features = config["features"]
target = config["target"]

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

# Calculate evaluation metrics
metrics = {
    "MSE": mean_squared_error(actual, predicted),
    "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
    "MAE": mean_absolute_error(actual, predicted),
    "MAPE": np.mean(np.abs((actual - predicted) / actual)) * 100,
    "R2": r2_score(actual, predicted)
}

# Save additional results
results = {
    "loss": history.history['loss'],
    "val_loss": history.history['val_loss'],
    "metrics": metrics,
    "predictions": predicted.flatten().tolist(),
    "actual": actual.flatten().tolist(),
    "comparison": comparison.tolist()
}

# Save the model and results in MLflow folder
model_path = os.path.join(config["model_save_path"], "lstm_model.h5")
save_model(model, model_path)

metrics_path = os.path.join(config["model_save_path"], "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=4)

comparison_csv_path = os.path.join(config["model_save_path"], "comparison.csv")
comparison_df.to_csv(comparison_csv_path, index=False)

print("Model, metrics, and comparison saved in the MLFLOW/Model folder.")

# Log to MLflow
mlflow.set_experiment("LSTM BTC Prediction")
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "patience": config["patience"],
        "sequence_length": config["sequence_length"]
    })

    # Log metrics
    mlflow.log_metrics(metrics)

    # Log loss and val_loss as metrics
    for epoch, (loss, val_loss) in enumerate(zip(history.history['loss'], history.history['val_loss']), 1):
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Log predictions, actuals, and comparison as artifacts
    mlflow.log_artifact(metrics_path)  # Save metrics.json
    mlflow.log_artifact(comparison_csv_path)  # Save comparison.csv

    # Log the trained model
    mlflow.log_artifact(model_path)  # Save the model file

print("All results logged to MLflow.")
