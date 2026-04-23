# 005_mnist_minimal_pytorch

Minimal PyTorch MNIST classifier ported from `../ai_research/projects/dl_classify_mnist`.

This unit intentionally keeps only the useful core:

- compressed MNIST IDX files in `data/raw/`
- one standalone PyTorch script in `code/train_mnist.py`
- saved model output in `model/`

Run from `code/`:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\train_mnist.py --epochs 3 --batch-size 256
```

Quick smoke test:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\train_mnist.py --epochs 1 --train-limit 2048 --test-limit 1024 --batch-size 256
```
