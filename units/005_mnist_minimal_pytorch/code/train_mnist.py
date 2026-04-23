import argparse
import gzip
import json
import struct
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MEAN = 0.1307
STD = 0.3081


def read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image IDX magic {magic} in {path}")
        data = torch.frombuffer(bytearray(f.read(count * rows * cols)), dtype=torch.uint8)
    images = data.reshape(count, 1, rows, cols).float().div(255.0)
    return images.sub(MEAN).div(STD)


def read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label IDX magic {magic} in {path}")
        return torch.frombuffer(bytearray(f.read(count)), dtype=torch.uint8).long()


def load_split(data_dir: Path, split: str, limit: int | None) -> TensorDataset:
    prefix = "train" if split == "train" else "t10k"
    images = read_idx_images(data_dir / f"{prefix}-images-idx3-ubyte.gz")
    labels = read_idx_labels(data_dir / f"{prefix}-labels-idx1-ubyte.gz")
    if limit:
        images = images[:limit]
        labels = labels[:limit]
    return TensorDataset(images, labels)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / total


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_path)

    train_ds = load_split(data_dir, "train", args.train_limit)
    test_ds = load_split(data_dir, "test", args.test_limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = MLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    start = time.perf_counter()
    final_loss = 0.0
    examples_seen = 0
    for _ in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            examples_seen += int(y.numel())

    elapsed_s = time.perf_counter() - start
    test_accuracy = evaluate(model, test_loader, device)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    metrics = {
        "epochs": args.epochs,
        "train_examples": len(train_ds),
        "test_examples": len(test_ds),
        "examples_seen": examples_seen,
        "elapsed_s": round(elapsed_s, 3),
        "examples_per_s": round(examples_seen / elapsed_s, 1),
        "final_train_loss": round(final_loss, 6),
        "test_accuracy": round(test_accuracy, 6),
        "device": str(device),
        "model_path": str(model_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/raw")
    parser.add_argument("--model-path", default="../model/mnist_linear.pt")
    parser.add_argument("--metrics-path", default="../model/metrics.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.train_limit = args.train_limit or None
    args.test_limit = args.test_limit or None
    return args


if __name__ == "__main__":
    train(parse_args())
