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


class LinearClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, hidden_size: int, hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        input_size = 28 * 28
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "linear":
        return LinearClassifier()
    if args.model == "deep-mlp":
        return DeepMLP(args.hidden_size, args.hidden_layers)
    raise ValueError(f"Unknown model: {args.model}")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        pred = logits.argmax(dim=1)
        loss_total += float(loss.item()) * int(y.numel())
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / total, loss_total / total


def train(args: argparse.Namespace) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_path)

    train_ds = load_split(data_dir, "train", args.train_limit)
    test_ds = load_split(data_dir, "test", args.test_limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    synchronize_if_needed(device)
    start = time.perf_counter()
    final_loss = 0.0
    examples_seen = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss_total = 0.0
        epoch_examples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            batch_examples = int(y.numel())
            examples_seen += batch_examples
            epoch_examples += batch_examples
            epoch_loss_total += final_loss * batch_examples

        synchronize_if_needed(device)
        val_accuracy, val_loss = evaluate(model, test_loader, device, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(epoch_loss_total / epoch_examples, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 6),
                "train_elapsed_s": round(time.perf_counter() - epoch_start, 3),
            }
        )

    synchronize_if_needed(device)
    elapsed_s = time.perf_counter() - start
    test_accuracy = history[-1]["val_accuracy"]

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
        "history": history,
        "device": str(device),
        "model": args.model,
        "hidden_size": args.hidden_size if args.model == "deep-mlp" else 0,
        "hidden_layers": args.hidden_layers if args.model == "deep-mlp" else 0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model", default="linear", choices=["linear", "deep-mlp"])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--hidden-layers", type=int, default=3)
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
