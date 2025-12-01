# data/lstm_hybrid.py

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMHybrid(nn.Module):
    """
    Basit bir LSTM tabanlı ikili sınıflandırıcı.
    Girdi: (batch, seq_len, input_size)
    Çıktı: p(y=1) (sigmoid)
    """
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)          # out: (B, T, H)
        last_hidden = out[:, -1, :]    # son time-step
        logit = self.fc(last_hidden)   # (B, 1)
        return logit.squeeze(-1)       # (B,)


def _build_sequence_dataset(
    clean_df,
    feature_cols: List[str],
    y_series,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    clean_df + y_series'den (N, seq_len, F) ve (N,) label üretir.
    Sadece y=0/1 kullandığımızı varsayıyoruz.
    """
    df = clean_df.reset_index(drop=True)
    y = y_series.reset_index(drop=True).astype(int)

    X = df[feature_cols].astype(float).values
    n_samples, n_features = X.shape

    if n_samples <= seq_len + 10:
        raise RuntimeError(f"LSTM için yeterli örnek yok: n_samples={n_samples}, seq_len={seq_len}")

    seqs = []
    labels = []
    # seq_len-1 .. n_samples-1
    for i in range(seq_len - 1, n_samples):
        seq = X[i - seq_len + 1 : i + 1]
        seqs.append(seq)
        labels.append(y.iloc[i])

    X_seqs = np.stack(seqs, axis=0).astype("float32")  # (N, T, F)
    y_arr = np.array(labels, dtype="float32")          # (N,)
    return X_seqs, y_arr


def train_lstm_hybrid(
    clean_df,
    feature_cols: List[str],
    y_series,
    interval: str,
    side: str,
    mode: str,
    model_dir: str,
    seq_len: int = 50,
    batch_size: int | None = None,
    device: str | None = None,
) -> str:
    """
    Basit LSTM offline eğitimi.
    - clean_df: anomaly filter sonrası veri (pandas DataFrame)
    - feature_cols: LSTM'e girecek feature kolonları (numeric)
    - y_series: 0/1 target (y_long veya y_short)
    - interval: '1m', '5m', ...
    - side: 'long' / 'short'
    - mode: shallow / full / deep -> epoch sayısını buradan seçeceğiz
    - model_dir: kaydetme klasörü
    - seq_len: son kaç bar kullanılacak (varsayılan 50)
    """
    os.makedirs(model_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mode'a göre epoch sayısı
    if mode == "deep":
        n_epochs = 5
    elif mode == "full":
        n_epochs = 4
    else:
        n_epochs = 3

    if batch_size is None:
        batch_size = 256

    print(
        f"[LSTM][{interval}][{side}] Eğitim başlıyor | "
        f"seq_len={seq_len}, epochs={n_epochs}, batch_size={batch_size}, device={device}"
    )

    # Dataset hazırla
    X_seqs, y_arr = _build_sequence_dataset(
        clean_df=clean_df,
        feature_cols=feature_cols,
        y_series=y_series,
        seq_len=seq_len,
    )

    # Basit global standardizasyon (feature bazlı)
    # (N, T, F) -> (N*T, F) üzerinden mean/std
    N, T, F = X_seqs.shape
    flat = X_seqs.reshape(-1, F)
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True) + 1e-8
    X_seqs = ((flat - mean) / std).reshape(N, T, F)

    X_tensor = torch.from_numpy(X_seqs)
    y_tensor = torch.from_numpy(y_arr)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LSTMHybrid(input_size=F, hidden_size=32, num_layers=1, dropout=0.1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            n_seen += bs

        avg_loss = total_loss / max(1, n_seen)
        print(f"[LSTM][{interval}][{side}] Epoch {epoch}/{n_epochs} - loss={avg_loss:.4f}")

    # Modeli kaydet
    model_path = os.path.join(model_dir, f"lstm_{interval}_{side}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": F,
            "hidden_size": 32,
            "seq_len": seq_len,
            "feature_cols": feature_cols,
        },
        model_path,
    )
    print(f"[LSTM][{interval}][{side}] Model kaydedildi: {model_path}")
    return model_path
