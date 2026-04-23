# STATUS.md

## Purpose

- Preserve the old MNIST classification idea as a tiny, rerunnable PyTorch example without porting the old Lightning/Hydra/W&B stack.

## Contents

- `code/train_mnist.py`: standalone PyTorch linear classifier over MNIST IDX gzip files.
- `data/raw/`: compressed MNIST files copied from `../ai_research/data/MNIST/raw`.
- `model/`: destination for trained checkpoint and metrics artifacts.
- `plot/cpu_vs_amd_gpu.svg`: CPU/GPU benchmark comparison.
- `plot/deep_mlp_learning_curve.svg`: 5-epoch deep-MLP train/validation curve from the AMD GPU run.

## Port Notes

- Source project: `../ai_research/projects/dl_classify_mnist`.
- Original setup used Lightning, Hydra configs, W&B media hooks, and `torchvision.datasets.MNIST`.
- Port keeps the same basic model shape from `task/fnn.yaml`: flattened 784-pixel input to 10 output classes.

## Verification

- Smoke test verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe`.
- Command: `python .\train_mnist.py --epochs 1 --train-limit 2048 --test-limit 1024 --batch-size 256`
- Result: `test_accuracy=0.69043`, `final_train_loss=1.13252`, `examples_per_s=42273.7` on CPU.
- Artifacts: `model/mnist_linear.pt` and `model/metrics.json`.
- GPU smoke test verified on April 23, 2026 on the AMD Radeon RX 7800 XT machine using PyTorch `2.9.1+rocm7.2.1`; PyTorch exposes this ROCm GPU through `device=cuda`.
- GPU command: `python .\train_mnist.py --device cuda --epochs 1 --train-limit 2048 --test-limit 1024 --batch-size 256 --metrics-path ..\model\metrics_cuda_smoke.json --model-path ..\model\mnist_linear_cuda_smoke.pt`
- GPU result: `test_accuracy=0.69043`, `final_train_loss=1.13252`, `examples_per_s=2609.9`. The tiny smoke test is slower on GPU because launch overhead dominates.
- The script now defaults to `--device auto`; on the AMD ROCm machine this resolves to `cuda`.
- Larger deep-MLP benchmark verified on April 23, 2026 with 20,000 train examples, 5,000 test examples, batch size 1024, 1 epoch, hidden size 2048, 3 hidden layers, and 10,020,874 parameters.
- Deep-MLP CPU result: `test_accuracy=0.9002`, `examples_per_s=482.8`, elapsed `41.429s`.
- Deep-MLP AMD GPU result: `test_accuracy=0.9002`, `examples_per_s=14056.4`, elapsed `1.423s`.
- Deep-MLP GPU speedup: `29.11x` by throughput.
- 5-epoch AMD GPU learning run reached validation accuracy `0.9538`; validation accuracy by epoch was `0.9002`, `0.9348`, `0.9510`, `0.9522`, `0.9538`.
- Plot regeneration verified with `python .\plot_metrics.py`.

## Next Steps

- Run the full 3-epoch local baseline if this unit becomes relevant again.
