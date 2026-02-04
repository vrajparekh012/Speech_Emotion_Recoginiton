import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Build CSV from CREMA-D (or similar datasets)
# -----------------------------
def build_csv_from_folder(base_dir, out_csv="dataset.csv"):
    # CREMA-D emotion labels
    emotion_map = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fearful",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad"
    }

    data = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".wav"):
                parts = f.split("_")
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    if emotion_code in emotion_map:
                        label = emotion_map[emotion_code]
                        data.append([os.path.join(root, f), label])

    df = pd.DataFrame(data, columns=["audio_path", "label"])
    df.to_csv(out_csv, index=False)
    print(f"CSV saved to {out_csv} with {len(df)} samples")
    return df

# -----------------------------
# Step 2: Dataset Loader
# -----------------------------
class SERDataset(Dataset):
    def __init__(self, df, sr=16000, n_mfcc=40, max_len=5.0, label_to_id=None):
        self.df = df
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.label_to_id = label_to_id or {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['audio_path']
        label = self.label_to_id[row['label']]

        y, sr = librosa.load(path, sr=self.sr)
        target_len = int(self.max_len * self.sr)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = torch.tensor(mfcc, dtype=torch.float).unsqueeze(0)
        return mfcc, label

# -----------------------------
# Step 3: Model
# -----------------------------
class SERModel(nn.Module):
    def __init__(self, n_mfcc=40, hidden_size=128, num_classes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.lstm = nn.LSTM(
            input_size=(n_mfcc // 4) * 32,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b, c, f, t = x.size()
        x = self.conv(x)
        b, c2, f2, t2 = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t2, c2 * f2)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.classifier(out)

# -----------------------------
# Step 4: Training
# -----------------------------
def train_ser(train_df, val_df, epochs=20, batch_size=32, lr=1e-3, device="cuda"):
    label_to_id = {label: idx for idx, label in enumerate(sorted(train_df['label'].unique()))}
    train_ds = SERDataset(train_df, label_to_id=label_to_id)
    val_ds = SERDataset(val_df, label_to_id=label_to_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(label_to_id)
    model = SERModel(n_mfcc=train_ds.n_mfcc, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X, y = X.to(device), torch.tensor(y).to(device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), torch.tensor(yv).to(device)
                logits = model(Xv)
                preds = logits.argmax(dim=1)
                correct += (preds == yv).sum().item()
                total += yv.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1} - Val Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "ser_best.pth")
    return best_acc, label_to_id

# -----------------------------
# Step 5: Inference
# -----------------------------
def infer_ser(audio_path, model_path="ser_best.pth", sr=16000, n_mfcc=40, device="cpu", label_to_id=None):
    id_to_label = {v: k for k, v in label_to_id.items()}
    model = SERModel(n_mfcc=n_mfcc, num_classes=len(label_to_id))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    y, _ = librosa.load(audio_path, sr=sr)
    target_len = int(5.0 * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = torch.tensor(mfcc, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mfcc)
        pred_id = logits.argmax(dim=1).item()
        return id_to_label[pred_id]

# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    # ✅ Your dataset folder (CREMA-D or similar)
    dataset_path = r"C:\Users\vrajp\Downloads\archive\AudioWAV"

    df = build_csv_from_folder(dataset_path, "dataset.csv")

    if len(df) == 0:
        raise ValueError("No audio files found! Check dataset path.")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    best_acc, label_to_id = train_ser(
        train_df, val_df,
        epochs=30,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Best Validation Accuracy:", best_acc)

    test_file = df.iloc[0]['audio_path']
    emotion = infer_ser(test_file, label_to_id=label_to_id)
    print(f"Predicted emotion for {test_file}: {emotion}")