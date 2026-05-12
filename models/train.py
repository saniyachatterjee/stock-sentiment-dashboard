import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# ── paths ──────────────────────────────────────────────────
MASTER_PATH  = Path("data/processed/master.parquet")
MODEL_DIR    = Path("models/saved")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── config ─────────────────────────────────────────────────
SEQUENCE_LEN = 20        # use last 20 days to predict next day
FEATURES = [
    "close", "open", "high", "low", "volume",
    "sentiment_score", "sentiment_3d_avg",
    "ma_5", "ma_20", "rsi", "momentum", "volatility",
    "daily_return", "return_5d",
    "bullish_count", "bearish_count"
]
TARGET = "close"
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 0.001
HIDDEN_SIZE = 64

# ── LSTM model definition ───────────────────────────────────
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # use last timestep

# ── data preparation ────────────────────────────────────────
def prepare_data(df, ticker):
    df_t = df[df["ticker"] == ticker].sort_values("date").copy()
    df_t = df_t[FEATURES].dropna()

    # scale features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_t)

    # scale target separately so we can inverse transform predictions
    target_scaler = MinMaxScaler()
    target_idx = FEATURES.index(TARGET)
    target_scaler.fit(df_t[[TARGET]])

    # create sequences
    X, y = [], []
    for i in range(SEQUENCE_LEN, len(scaled)):
        X.append(scaled[i - SEQUENCE_LEN:i])
        y.append(scaled[i, target_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 80/20 split — NO shuffle for time series
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler, target_scaler, df_t

# ── training loop ───────────────────────────────────────────
def train_model(X_train, y_train, input_size):
    model = StockLSTM(input_size=input_size, hidden_size=HIDDEN_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train)
    y_t = torch.tensor(y_train).unsqueeze(1)

    best_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_t.size(0))
        epoch_loss = 0

        for i in range(0, X_t.size(0), BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            xb, yb = X_t[idx], y_t[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / (len(X_train) / BATCH_SIZE)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.6f}")

    model.load_state_dict(best_state)
    return model

# ── evaluation ──────────────────────────────────────────────
def evaluate(model, X_test, y_test, target_scaler):
    model.eval()
    with torch.no_grad():
        X_t    = torch.tensor(X_test)
        preds  = model(X_t).numpy().flatten()

    # inverse transform to real prices
    preds_real  = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actual_real = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(actual_real, preds_real)
    rmse = np.sqrt(mean_squared_error(actual_real, preds_real))

    # directional accuracy
    actual_dir = np.diff(actual_real) > 0
    pred_dir   = np.diff(preds_real)  > 0
    dir_acc    = np.mean(actual_dir == pred_dir) * 100

    return mae, rmse, dir_acc, preds_real, actual_real

# ── main ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading master data...")
    master = pd.read_parquet(MASTER_PATH)

    results = {}

    for ticker in ["AAPL", "MSFT", "TSLA"]:
        print(f"\n{'='*40}")
        print(f"Training LSTM for {ticker}...")
        print(f"{'='*40}")

        X_train, X_test, y_train, y_test, \
        scaler, target_scaler, df_t = prepare_data(master, ticker)

        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

        model = train_model(X_train, y_train, input_size=len(FEATURES))

        mae, rmse, dir_acc, preds, actuals = evaluate(
            model, X_test, y_test, target_scaler
        )

        print(f"\n--- {ticker} Results ---")
        print(f"MAE:                  ${mae:.2f}")
        print(f"RMSE:                 ${rmse:.2f}")
        print(f"Directional Accuracy: {dir_acc:.1f}%")

        # save model + scalers
        torch.save(model.state_dict(),
                   MODEL_DIR / f"{ticker}_lstm.pt")
        with open(MODEL_DIR / f"{ticker}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(MODEL_DIR / f"{ticker}_target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)

        # save predictions for dashboard
        pred_df = pd.DataFrame({
            "actual": actuals,
            "predicted": preds
        })
        pred_df.to_csv(MODEL_DIR / f"{ticker}_predictions.csv", index=False)

        results[ticker] = {
            "mae": mae, "rmse": rmse, "dir_acc": dir_acc
        }

    print(f"\n{'='*40}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*40}")
    for ticker, r in results.items():
        print(f"{ticker}: MAE=${r['mae']:.2f} | "
              f"RMSE=${r['rmse']:.2f} | "
              f"Directional Acc={r['dir_acc']:.1f}%")

    print("\nAll models saved to models/saved/")