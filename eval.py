import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, resnet34
from utils import accuracy

def get_loader(dataset, data_dir, batch_size, num_workers):
    if dataset=="mnist":
        mean, std = (0.1307,), (0.3081,)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        ds = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)
        in_ch=1; num_classes=10
    elif dataset=="cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf)
        in_ch=3; num_classes=10
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tf)
        in_ch=3; num_classes=100
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True), in_ch, num_classes

def build_model(model_name, in_ch, num_classes):
    if model_name=="resnet18":
        m = resnet18(num_classes=num_classes)
    elif model_name=="resnet34":
        m = resnet34(num_classes=num_classes)
    elif model_name=="simplecnn":
        import torch.nn as nn
        m = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7 if in_ch==1 else 64*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError
    if in_ch==1 and model_name.startswith("resnet"):
        import torch.nn as nn
        m.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","cifar10","cifar100"], required=True)
    ap.add_argument("--model", choices=["simplecnn","resnet18","resnet34"], required=True)
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--ckpt", required=True, help="path to checkpoint .pt (best.pt or epoch_*.pt)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader, in_ch, num_classes = get_loader(args.dataset, args.data_dir, args.batch_size, args.num_workers)
    model = build_model(args.model, in_ch, num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            acc1 = accuracy(logits, y, topk=(1,))[0]
            val_loss += loss.item() * x.size(0)
            val_acc  += acc1 * x.size(0) / 100.0
    val_loss /= len(loader.dataset)
    val_acc  = 100.0 * val_acc / len(loader.dataset)
    print(f"[eval] loss={val_loss:.4f} acc={val_acc:.2f}%  ckpt={args.ckpt}")

if __name__ == "__main__":
    main()
