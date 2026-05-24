import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .config import SystemConfig
from .data_simulator import SyntheticOlfactoryDataset
from .models import FusionNet
from .utils import seed_everything, ensure_dir


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        time_signal = batch["time_signal"].to(device)
        spectral_image = batch["spectral_image"].to(device)
        env = batch["env"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits, _ = model(time_signal, spectral_image, env)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        time_signal = batch["time_signal"].to(device)
        spectral_image = batch["spectral_image"].to(device)
        env = batch["env"].to(device)
        label = batch["label"].to(device)

        logits, _ = model(time_signal, spectral_image, env)
        loss = criterion(logits, label)

        total_loss += loss.item() * label.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    return total_loss / total, correct / total


def main():
    cfg = SystemConfig()
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = SyntheticOlfactoryDataset(cfg, num_samples=1200)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

    model = FusionNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ensure_dir(os.path.dirname(cfg.checkpoint_path))
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": cfg.class_names,
                    "best_val_acc": best_val_acc,
                },
                cfg.checkpoint_path,
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model to: {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
