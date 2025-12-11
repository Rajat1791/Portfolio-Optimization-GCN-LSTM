import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from scipy.optimize import minimize
# --- FUNCTIONS ---
def fetch_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True)
    data = data['Close'].stack().reset_index()
    data.columns = ['Date', 'Stock', 'Close']
    data.set_index(['Date', 'Stock'], inplace=True)
    return data

def compute_technical_indicators(stock_data):
    all_stocks = []
    for stock in stock_data.index.get_level_values(1).unique():
        stock_df = stock_data.xs(stock, level=1, drop_level=False).copy()
        delta = stock_df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=stock_df.index).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss, index=stock_df.index).rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        stock_df['RSI'] = 100 - (100 / (1 + rs))
        ema_12 = stock_df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = stock_df['Close'].ewm(span=26, adjust=False).mean()
        stock_df['MACD'] = ema_12 - ema_26
        sma_20 = stock_df['Close'].rolling(window=20).mean()
        std_20 = stock_df['Close'].rolling(window=20).std()
        stock_df['Upper_BB'] = sma_20 + (2 * std_20)
        stock_df['Lower_BB'] = sma_20 - (2 * std_20)
        all_stocks.append(stock_df)
    return pd.concat(all_stocks).dropna()

def fetch_fundamental_data(symbols):
    fundamental_data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            fundamental_data[symbol] = {
                "P/E Ratio": info.get("trailingPE", np.nan),
                "P/B Ratio": info.get("priceToBook", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "EPS": info.get("trailingEps", np.nan),
                "Debt-to-Equity": info.get("debtToEquity", np.nan)
            }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            fundamental_data[symbol] = {k: np.nan for k in ["P/E Ratio", "P/B Ratio", "ROE", "EPS", "Debt-to-Equity"]}
    df = pd.DataFrame.from_dict(fundamental_data, orient='index')
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
        df[col] = np.where(df[col] > df[col].quantile(0.99), df[col].quantile(0.99), df[col])
    return df

def preprocess(stock_data):
    features = ['Close', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB', 'P/E Ratio', 'P/B Ratio', 'ROE', 'EPS', 'Debt-to-Equity']
    stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
    scaler = MinMaxScaler()
    stock_data[features] = scaler.fit_transform(stock_data[features])
    return stock_data

def create_lstm_sequences(data, sequence_length=5):
    X, Y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        Y.append(data[i+sequence_length])
    return np.array(X), np.array(Y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# --- MAIN EXECUTION ---
start_date = datetime.date.today().replace(year=datetime.date.today().year - 10)
end_date = datetime.date.today()
companies = ['INFY.NS', 'TCS.NS', 'TATAMOTORS.NS', 'MARUTI.NS', 'SUNPHARMA.NS',
             'CIPLA.NS', 'ITC.NS', 'MARICO.NS', 'BHARTIARTL.NS', 'BAJAJ-AUTO.NS',
             'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'KOTAKBANK.NS',
             'HINDUNILVR.NS', 'ASIANPAINT.NS', 'NESTLEIND.NS', 'LT.NS', 'ULTRACEMCO.NS',
             'BAJFINANCE.NS', 'SBIN.NS', 'TECHM.NS', 'WIPRO.NS', 'HCLTECH.NS',
             'POWERGRID.NS', 'NTPC.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'DRREDDY.NS']

stock_data = fetch_stock_data(companies, start_date, end_date)
stock_data = compute_technical_indicators(stock_data)
nonnormal_data = stock_data.reset_index().copy()
fundamental_data = fetch_fundamental_data(companies)
stock_data_reset = stock_data.reset_index()
merged_data = stock_data_reset.merge(fundamental_data, left_on="Stock", right_index=True, how="left")
merged_data = merged_data.apply(pd.to_numeric, errors='ignore')
merged_data = preprocess(merged_data)

# Prepare LSTM sequences
features = merged_data.drop(columns=['Date', 'Stock']).values
labels = merged_data['Close'].values

X_lstm, _ = create_lstm_sequences(features, sequence_length=5)
_, Y_lstm = create_lstm_sequences(labels.reshape(-1, 1), sequence_length=5)
Y_lstm = Y_lstm.flatten()

X_tensor = torch.tensor(X_lstm, dtype=torch.float)
Y_tensor = torch.tensor(Y_lstm, dtype=torch.float)

train_len = int(0.8 * len(X_tensor))
X_train = X_tensor[:train_len]
Y_train = Y_tensor[:train_len]
X_val = X_tensor[train_len:]
Y_val = Y_tensor[train_len:]

model = LSTMModel(input_size=X_tensor.shape[2], hidden_size=64, output_size=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train).flatten()
    loss = loss_fn(output, Y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_output = model(X_val).flatten()
        val_loss = loss_fn(val_output, Y_val)
        val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_tensor).flatten().numpy()

true_values = Y_tensor.numpy()

plt.figure(figsize=(10, 6))
plt.plot(true_values, label="True Prices", color='blue')
plt.plot(predictions, label="Predicted Prices", color='red', linestyle='--')
plt.title("LSTM Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss", color='blue')
plt.plot(val_losses, label="Val Loss", color='orange')
plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

mse = mean_squared_error(true_values, predictions)
mae = mean_absolute_error(true_values, predictions)
r2 = r2_score(true_values, predictions)
medae = median_absolute_error(true_values, predictions)

print("\n📈 LSTM Model Evaluation Metrics:")
print(f"  MSE      : {mse:.4f}")
print(f"  MAE      : {mae:.4f}")
print(f"  R²       : {r2:.4f}")
print(f"  Median AE: {medae:.4f}")

# --- Per-Stock Evaluation (approximate via labels from merged_data) ---
merged_data = merged_data.reset_index(drop=True)
stock_labels = merged_data['Stock'].values[5:]  # Offset due to LSTM sequence length
metrics = []
unique_stocks = np.unique(stock_labels)
for stock in unique_stocks:
    stock_mask = stock_labels == stock
    y_true = true_values[stock_mask]
    y_pred = predictions[stock_mask]
    if len(y_true) == 0: continue
    mse_s = mean_squared_error(y_true, y_pred)
    mae_s = mean_absolute_error(y_true, y_pred)
    r2_s = r2_score(y_true, y_pred)
    medae_s = median_absolute_error(y_true, y_pred)
    metrics.append({
        'Stock': stock,
        'MSE': mse_s,
        'MAE': round(mae_s, 4),
        'R²': round(r2_s, 4),
        'Median_AE': medae_s
    })
metrics_df = pd.DataFrame(metrics)
print("\n📊 Evaluation Metrics Per Stock:")
print(metrics_df.to_string(index=False))

# --- K-FOLD CROSS-VALIDATION ---
X = X_tensor.numpy()
y = Y_tensor.numpy()
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")
    train_x = torch.tensor(X[train_idx], dtype=torch.float)
    train_y = torch.tensor(y[train_idx], dtype=torch.float)
    val_x = torch.tensor(X[val_idx], dtype=torch.float)
    val_y = torch.tensor(y[val_idx], dtype=torch.float)

    model = LSTMModel(input_size=X.shape[2], hidden_size=64, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(train_x).flatten()
        loss = loss_fn(output, train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(val_x).flatten()
            val_loss = loss_fn(val_output, val_y)
            val_losses.append(val_loss.item())
        model.train()

    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(f"Fold {fold+1} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    model.eval()
    with torch.no_grad():
        val_pred = model(val_x).flatten().numpy()
        val_true = val_y.numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(val_true, label='True Prices', alpha=0.7)
    plt.plot(val_pred, label='Predictions', linestyle='--', alpha=0.9)
    plt.title(f"Fold {fold+1} Prediction Comparison")
    plt.xlabel("Validation Samples")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.show()

    mse_f = mean_squared_error(val_true, val_pred)
    mae_f = mean_absolute_error(val_true, val_pred)
    medae_f = median_absolute_error(val_true, val_pred)
    r2_f = r2_score(val_true, val_pred)

    print(f"Fold {fold+1} Metrics:")
    print(f"  MSE: {mse_f:.4f}")
    print(f"  MAE: {mae_f:.4f}")
    print(f"  Median AE: {medae_f:.4f}")
    print(f"  R²: {r2_f:.4f}")

    fold_metrics.append({
        'Fold': fold+1,
        'MSE': mse_f,
        'MAE': mae_f,
        'Median_AE': medae_f,
        'R2': r2_f
    })

metrics_ev = pd.DataFrame(fold_metrics)
print("\n📊 Evaluation Metrics Per Fold:")
print(metrics_ev.to_string(index=False))
