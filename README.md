# nn-benchmarks-baselines

Vanilla PyTorch training loops for MNIST / CIFAR-10 / CIFAR-100 with ResNet and a simple CNN.  
Includes CSV logging, checkpoints, AMP (mixed precision), schedulers, and plots. No proprietary logic.

## Quickstart (Windows PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --dataset cifar10 --model resnet18 --epochs 20 --augment --amp --out runs/cifar10_resnet18
python plots.py --log runs/cifar10_resnet18/log.csv
python eval.py --dataset cifar10 --model resnet18 --ckpt runs/cifar10_resnet18/checkpoints/best.pt
