# STATUS.md

## Purpose

- Preserve the old MNIST classification idea as a tiny, rerunnable PyTorch example without porting the old Lightning/Hydra/W&B stack.

## Contents

- `code/train_mnist.py`: standalone PyTorch linear classifier over MNIST IDX gzip files.
- `data/raw/`: compressed MNIST files copied from `../ai_research/data/MNIST/raw`.
- `model/`: destination for trained checkpoint and metrics artifacts.

## Port Notes

- Source project: `../ai_research/projects/dl_classify_mnist`.
- Original setup used Lightning, Hydra configs, W&B media hooks, and `torchvision.datasets.MNIST`.
- Port keeps the same basic model shape from `task/fnn.yaml`: flattened 784-pixel input to 10 output classes.

## Verification

- Smoke test verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe`.
- Command: `python .\train_mnist.py --epochs 1 --train-limit 2048 --test-limit 1024 --batch-size 256`
- Result: `test_accuracy=0.69043`, `final_train_loss=1.13252`, `examples_per_s=42273.7` on CPU.
- Artifacts: `model/mnist_linear.pt` and `model/metrics.json`.

## Next Steps

- Run the full 3-epoch local baseline if this unit becomes relevant again.
