import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd
from pathlib import Path
import os

class SwingTradingModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)
    
# Building out features intended for model training
def build_features(df: pd.DataFrame):
    df = df.copy()

    # Returns
    df["return_1d"] = df["Close"].pct_change(1)
    df["return_3d"] = df["Close"].pct_change(3)
    df["return_5d"] = df["Close"].pct_change(5)
    df["return_10d"] = df["Close"].pct_change(10)

    # Log returns
    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))

    # Simple Moving averages
    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = df["Close"].rolling(window).mean()
        df[f"price_sma_ratio_{window}"] = df["Close"] / df[f"sma_{window}"] # Above 1, price is elevated, below price is depressed

    # Volatility
    for window in [5, 10, 20]:
        df[f"vol_{window}"] = df["Close"].rolling(window).std()

    # True range / ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # Volume features
    df["vol_change"] = df["Volume"].pct_change(1)
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]

    # Forward
    df[f"fwd_return_5d"] = df["Close"].shift(-5) / df["Close"] - 1
    
    return df

def load_or_fetch_raw(symbol, start, end, data_dir="data/raw"):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(data_dir) / f"{symbol}_raw.csv"

    if file_path.exists():
        print("Loading data...")
        return pd.read_csv(file_path, parse_dates=["Date"])

    print("Fetching raw data...")
    df = yf.download(symbol, start=start, end=end, auto_adjust=False)
    df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    df.to_csv(file_path, index=False)
    return df


# Exports features to CSV. Uses load_or_fetch_raw for getting raw data, then builds features and saves.
def export_features_csv(symbol, start, end, data_dir="data"):
    Path(data_dir).mkdir(exist_ok=True)
    file_path = Path(data_dir) / f"{symbol}_features.csv"

    if file_path.exists():
        print("Feature CSV already exists.")
        return pd.read_csv(file_path, parse_dates=["Date"])

    # print("Building feature dataset...")
    df = load_or_fetch_raw(symbol, start, end)

    df = build_features(df)
    df.dropna(inplace=True)

    df.to_csv(file_path, index=False)
    # print(f"Saved features to {file_path}")
    return df

def dataframe_to_tensors(df, feature_cols, target_col):
    X_np = df[feature_cols].values.astype(np.float32)
    y_np = df[target_col].values.astype(np.float32)

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np).unsqueeze(1)  # shape (N, 1)

    return X, y

def train_val_split(df, val_fraction=0.2):
    split_idx = int(len(df) * (1 - val_fraction))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    return train_df, val_df

def main():
    df = export_features_csv(symbol="MSFT", start="2010-01-01", end="2020-12-31")

    # print(df.shape)
    # print(df.columns)
    df.dropna(inplace=True)
    FEATURE_COLUMNS = [
        "return_1d", "return_3d", "return_5d", "return_10d",
        "log_return_1d",
        "price_sma_ratio_5", "price_sma_ratio_10",
        "price_sma_ratio_20", "price_sma_ratio_50",
        "vol_5", "vol_10", "vol_20",
        "atr_14",
        "vol_change", "vol_ratio"
    ]
    # We want to split our train and validation sets based on time
    # We do not want information from the future leaking into the past
    # This also gives the model the chance to learn patterns that develop over time
    
    train_df, val_df = train_val_split(df)
    
    X_train, Y_train = dataframe_to_tensors(
        train_df,
        FEATURE_COLUMNS,
        target_col="fwd_return_5d"
    )
    
    # print(X_train.shape, Y_train.shape)
    # print(train_df["Date"].min(), train_df["Date"].max())
    # print(val_df["Date"].min(), val_df["Date"].max())
    
    
    # X = data
    # Y = other_data
    # X = torch.tensor(X, dtype=torch.float32)
    # Y = torch.tensor(Y, dtype = torch.float32)
    
    # model = SwingTradingModel(input_size = X.shape[1])
    
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    
    
    # for epoch in range(100):
    #     model.train()
    #     optimizer.zero_grad()
    #     outputs = model(X.to(device))
    #     loss = criterion(outputs, Y.to(device))
    #     loss.backward()
    #     optimizer.step()
        
    #     if (epoch+1) % 10 == 0:
    #         print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    
    
    
if __name__ == "__main__":
    main()