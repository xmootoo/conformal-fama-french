from models import train_price_forecaster, predict
from dataloading import load_forecasting_data, temporal_train_test_split
import numpy as np


factors = "5_factor"
symbol = "AAPL"
seq_len = 365 # (Look back 1 year)
pred_len = 30 # (Look ahead 30 days)
X, y = load_forecasting_data(symbol, factors, seq_len, pred_len)

X_train, y_train, X_val, y_val, X_test, y_test = temporal_train_test_split(X, y, train_ratio=0.7, val_ratio=0.15, shuffle=True, seed=42)

model, scaler = train_price_forecaster(X_train, y_train)

# Add conformal prediction here with validation set
y_hat = predict(model, scaler, X_test)

# Evaluate MSE. The average error in Daily Errors (%)
mse_loss = np.mean((y_hat - y_test) ** 2)*100
print(f"MSE: {mse_loss:.2f}%")
