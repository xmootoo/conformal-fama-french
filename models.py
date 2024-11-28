import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def train_price_forecaster(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Trains a linear regression model to forecast future prices.

    Args:
        X: Shape (n_samples, seq_len, n_features)
        y: Shape (n_samples, pred_len)

    Returns:
        tuple: (trained model, scaler)
    """
    n_samples, seq_len, n_features = X.shape

    # Reshape to 2D for scaling each feature independently
    X_reshaped = X.reshape(-1, n_features)  # Combine n_samples and seq_len

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)

    # Reshape back to 3D
    X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)

    # Reshape to 2D for LinearRegression (it expects 2D input)
    X_scaled = X_scaled.reshape(n_samples, seq_len * n_features)

    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler

# When predicting:
def predict(model, scaler, X_new):
    """
    Make predictions on new data.

    Args:
        X_new: Shape (n_samples, seq_len, n_features)
    """
    n_samples, seq_len, n_features = X_new.shape

    # Reshape and scale
    X_reshaped = X_new.reshape(-1, n_features)
    X_scaled = scaler.transform(X_reshaped)

    # Reshape for prediction
    X_scaled = X_scaled.reshape(n_samples, seq_len * n_features)

    return model.predict(X_scaled)
