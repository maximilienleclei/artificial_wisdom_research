# 005_mnist_minimal_pytorch

Minimal PyTorch MNIST classifier ported from `../ai_research/projects/dl_classify_mnist`.

This unit intentionally keeps only the useful core:

- compressed MNIST IDX files in `data/raw/`
- one standalone PyTorch script in `code/train_mnist.py`
- saved model/metrics output in `model/`
- benchmark plot in `plot/`

Run from `code/`:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\train_mnist.py --epochs 3 --batch-size 256
```

The script defaults to `--device auto`, which uses `cuda` when PyTorch exposes a GPU. On the AMD Radeon RX 7800 XT machine, the ROCm PyTorch build exposes the GPU through that CUDA device path.

Quick smoke test:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\train_mnist.py --epochs 1 --train-limit 2048 --test-limit 1024 --batch-size 256
```

Larger CPU/GPU benchmark:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\train_mnist.py --device cuda --model deep-mlp --hidden-size 2048 --hidden-layers 3 --epochs 1 --train-limit 20000 --test-limit 5000 --batch-size 1024
```

Regenerate the CPU/GPU comparison and learning-curve plots from saved metrics:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\plot_metrics.py
```
