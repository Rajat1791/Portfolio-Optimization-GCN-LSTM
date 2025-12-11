import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.model_selection import KFold
from scipy.optimize import minimize
import random
import torch
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Data Fetching and Feature Engineering  ---
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
    final_stock_data = pd.concat(all_stocks)
    return final_stock_data.dropna()

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
            print(f"⚠️ Error fetching data for {symbol}: {e}")
            fundamental_data[symbol] = {
                "P/E Ratio": np.nan, "P/B Ratio": np.nan,
                "ROE": np.nan, "EPS": np.nan, "Debt-to-Equity": np.nan
            }
    fundamental_df = pd.DataFrame.from_dict(fundamental_data, orient='index')
    for col in fundamental_df.columns:
        fundamental_df[col] = fundamental_df[col].fillna(fundamental_df[col].median())
    for col in ["P/E Ratio", "P/B Ratio", "ROE", "EPS", "Debt-to-Equity"]:
        upper_limit = fundamental_df[col].quantile(0.99)
        fundamental_df[col] = np.where(fundamental_df[col] > upper_limit, upper_limit, fundamental_df[col])
    return fundamental_df

def preprocess(stock_data):
    features = ['Close', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB', 'P/E Ratio', 'P/B Ratio', 'ROE', 'EPS', 'Debt-to-Equity']
    stock_data = stock_data.fillna(method='ffill').fillna(method='bfill')
    scaler = MinMaxScaler()
    stock_data[features] = scaler.fit_transform(stock_data[features])
    return stock_data

# --- Build real edge_index based on correlation ---
def build_correlation_edge_index(price_df, threshold=0.7):
    correlation_matrix = price_df.corr()
    stocks = list(correlation_matrix.columns)
    stock_to_idx = {stock: idx for idx, stock in enumerate(stocks)}
    edges = []
    for i, stock1 in enumerate(stocks):
        for j, stock2 in enumerate(stocks):
            if i != j and abs(correlation_matrix.loc[stock1, stock2]) >= threshold:
                edges.append([stock_to_idx[stock1], stock_to_idx[stock2]])
    if not edges:
        raise ValueError("No edges found at the given threshold.")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, stocks

# --- GCNModel, LSTMModel, GCN_LSTM_Model, create_gcn_data ---
class GCNModel(nn.Module):
    def __init__(self, input_features, hidden_channels, output_features):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_features)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class GCN_LSTM_Model(nn.Module):
    def __init__(self, gcn_model, lstm_model):
        super(GCN_LSTM_Model, self).__init__()
        self.gcn_model = gcn_model
        self.lstm_model = lstm_model
    def forward(self, x, edge_index):
        gcn_out = self.gcn_model(x, edge_index)
        gcn_out = gcn_out.view(-1, 1, gcn_out.size(1))
        lstm_out = self.lstm_model(gcn_out)
        return lstm_out

def create_gcn_data(stock_data, edge_index):
    features = stock_data.drop(columns=['Date', 'Stock']).values
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(stock_data['Close'].values, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

# --- MAIN PIPELINE STARTS HERE ---

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

# --- SPLIT BY DATE FOR PORTFOLIO OPTIMIZATION ---
all_dates = pd.to_datetime(merged_data['Date'].unique())
all_dates = np.sort(all_dates)
split_idx = int(len(all_dates) * 0.8)
train_end_date = all_dates[split_idx - 1]
test_start_date = all_dates[split_idx]
print(f"Train end date: {train_end_date}, Test start date: {test_start_date}")

train_mask = pd.to_datetime(merged_data['Date']) <= train_end_date
test_mask = pd.to_datetime(merged_data['Date']) > train_end_date
train_data = merged_data[train_mask].reset_index(drop=True)
test_data = merged_data[test_mask].reset_index(drop=True)

nonnormal_data = stock_data_reset.copy()
price_df = nonnormal_data.pivot(index='Date', columns='Stock', values='Close').sort_index()
price_df_train = price_df.loc[:str(train_end_date)]
price_df_test = price_df.loc[str(test_start_date):]
returns_df_train = price_df_train.pct_change().dropna()
returns_df_test = price_df_test.pct_change().dropna()

# --- Build real edge_index based on price correlation ---
edge_index, stocks = build_correlation_edge_index(price_df, threshold=0.7)
print(f"Number of edges in graph: {edge_index.shape[1]}")

# --- GCN-LSTM TRAINING AND EVALUATION ---
gcn_data = create_gcn_data(merged_data, edge_index)
gcn_model = GCNModel(input_features=gcn_data.x.shape[1], hidden_channels=64, output_features=32)
lstm_model = LSTMModel(input_size=32, hidden_size=64, output_size=1)
gcn_lstm_model = GCN_LSTM_Model(gcn_model, lstm_model)

optimizer = torch.optim.Adam(gcn_lstm_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

total_len = len(gcn_data.x)
train_len = int(total_len * 0.8)
train_x = gcn_data.x[:train_len]
train_y = gcn_data.y[:train_len]
val_x = gcn_data.x[train_len:]
val_y = gcn_data.y[train_len:]

train_losses = []
val_losses = []

gcn_lstm_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    output = gcn_lstm_model(train_x, gcn_data.edge_index)
    loss = loss_fn(output.flatten(), train_y)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    gcn_lstm_model.eval()
    with torch.no_grad():
        val_output = gcn_lstm_model(val_x, gcn_data.edge_index)
        val_loss = loss_fn(val_output.flatten(), val_y)
        val_losses.append(val_loss.item())

    gcn_lstm_model.train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Inference
gcn_lstm_model.eval()
with torch.no_grad():
    predictions = gcn_lstm_model(gcn_data.x, gcn_data.edge_index)
predictions = predictions.flatten().numpy()
true_values = gcn_data.y.numpy()

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(true_values, label="True Stock Prices", color='blue')
plt.plot(predictions, label="Predicted Stock Prices", color='red', linestyle='--')
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction (True vs Predicted)")
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss", color='blue')
plt.plot(val_losses, label="Validation Loss", color='orange', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.show()

# --- Overall Model Evaluation Metrics (Single Value for All Stocks) ---
overall_mse = mean_squared_error(true_values, predictions)
overall_mae = mean_absolute_error(true_values, predictions)
overall_r2 = r2_score(true_values, predictions)
overall_medae = median_absolute_error(true_values, predictions)

print("\n📈 Overall Model Evaluation Metrics:")
print(f"  MSE      : {overall_mse:.4f}")
print(f"  MAE      : {overall_mae:.4f}")
print(f"  R²       : {overall_r2:.4f}")
print(f"  Median AE: {overall_medae:.4f}")

# --- Evaluation Metrics per Stock ---
merged_data = merged_data.reset_index(drop=True)
stock_labels = merged_data['Stock'].values[:len(predictions)]

metrics = []
unique_stocks = np.unique(stock_labels)
for stock in unique_stocks:
    stock_mask = stock_labels == stock
    y_true = true_values[stock_mask]
    y_pred = predictions[stock_mask]
    if len(y_true) == 0: continue
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    med_ae = median_absolute_error(y_true, y_pred)
    metrics.append({
        'Stock': stock,
        'MSE': mse,
        'MAE': round(mae, 4),
        'R²': round(r2, 4),
        'Median_AE': med_ae
    })
metrics_df = pd.DataFrame(metrics)
print("\n📊 Evaluation Metrics Per Stock:")
print(metrics_df.to_string(index=False))

# --- K-FOLD CROSS-VALIDATION ---
X = gcn_data.x.numpy()
y = gcn_data.y.numpy()
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_metrics = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")
    train_x = torch.tensor(X[train_idx], dtype=torch.float)
    train_y = torch.tensor(y[train_idx], dtype=torch.float)
    val_x = torch.tensor(X[val_idx], dtype=torch.float)
    val_y = torch.tensor(y[val_idx], dtype=torch.float)
    gcn_model = GCNModel(input_features=X.shape[1], hidden_channels=64, output_features=32)
    lstm_model = LSTMModel(input_size=32, hidden_size=64, output_size=1)
    model = GCN_LSTM_Model(gcn_model, lstm_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(train_x, edge_index)
        loss = loss_fn(output.flatten(), train_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(val_x, edge_index)
            val_loss = loss_fn(val_output.flatten(), val_y)
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
        val_pred = model(val_x, edge_index).flatten().numpy()
        val_true = val_y.numpy()
    plt.figure(figsize=(10, 4))
    plt.plot(val_true, label='True Prices', alpha=0.7)
    plt.plot(val_pred, label='Predictions', linestyle='--', alpha=0.9)
    plt.title(f"Fold {fold+1} Prediction Comparison")
    plt.xlabel("Validation Samples")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.show()
    mse = mean_squared_error(val_true, val_pred)
    mae = mean_absolute_error(val_true, val_pred)
    med_ae = median_absolute_error(val_true, val_pred)
    r2 = r2_score(val_true, val_pred)
    print(f"Fold {fold+1} Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  Median AE: {med_ae:.4f}")
    print(f"  R²: {r2:.4f}")
    fold_metrics.append({
        'Fold': fold+1,
        'MSE': mse,
        'MAE': mae,
        'Median_AE': med_ae,
        'R2': r2
    })
metrics_ev = pd.DataFrame(fold_metrics)
print("\n📊 Evaluation Metrics Per Fold:")
print(metrics_ev.to_string(index=False))