import argparse, os, csv, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, resnet34

from utils import set_seed, accuracy, ensure_dir
from tqdm import tqdm

def get_dataloaders(dataset, data_dir, batch_size, num_workers, augment):
    if dataset.lower() == "mnist":
        mean, std = (0.1307,), (0.3081,)
        tf_train = [T.ToTensor(), T.Normalize(mean, std)]
        tf_test  = [T.ToTensor(), T.Normalize(mean, std)]
        if augment:
            # light augmentation: none needed for MNIST; keep as-is
            pass
        train = torchvision.datasets.MNIST(root=data_dir, train=True,  download=True, transform=T.Compose(tf_train))
        test  = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=T.Compose(tf_test))
        in_ch = 1; num_classes = 10
    elif dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        tf_train = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
        tf_train += [T.ToTensor(), T.Normalize(mean, std)]
        tf_test  = [T.ToTensor(), T.Normalize(mean, std)]
        train = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=T.Compose(tf_train))
        test  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=T.Compose(tf_test))
        in_ch = 3; num_classes = 10
    elif dataset.lower() == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        tf_train = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
        tf_train += [T.ToTensor(), T.Normalize(mean, std)]
        tf_test  = [T.ToTensor(), T.Normalize(mean, std)]
        train = torchvision.datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=T.Compose(tf_train))
        test  = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=T.Compose(tf_test))
        in_ch = 3; num_classes = 100
    else:
        raise ValueError("dataset must be one of: mnist, cifar10, cifar100")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, in_ch, num_classes

def build_model(model_name, in_ch, num_classes):
    if model_name == "resnet18":
        m = resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        m = resnet34(num_classes=num_classes)
    elif model_name == "simplecnn":
        m = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7 if in_ch==1 else 64*8*8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError("model must be one of: resnet18, resnet34, simplecnn")
    # If grayscale, expand to 3 channels for ResNet
    if in_ch == 1 and model_name.startswith("resnet"):
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist","cifar10","cifar100"], default="cifar10")
    ap.add_argument("--model",   choices=["simplecnn","resnet18","resnet34"], default="resnet18")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--opt", choices=["sgd","adam"], default="sgd")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--sched", choices=["cosine","step","none"], default="cosine")
    ap.add_argument("--step_size", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="use mixed precision")
    ap.add_argument("--out", default="./runs/cifar10_resnet18")
    ap.add_argument("--resume", default="", help="path to checkpoint to resume")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, in_ch, num_classes = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers, args.augment
    )
    model = build_model(args.model, in_ch, num_classes).to(device)

    if args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.sched == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.sched == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    ensure_dir(args.out)
    ensure_dir(os.path.join(args.out, "checkpoints"))

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        if scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[resume] epoch={start_epoch} best_val_acc={best_val_acc:.2f}%")

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # CSV logger
    log_path = os.path.join(args.out, "log.csv")
    if start_epoch == 0 and not Path(log_path).exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","lr"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc1 = accuracy(logits, y, topk=(1,))[0]
            train_loss += loss.item() * x.size(0)
            train_acc  += acc1 * x.size(0) / 100.0
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc1:.1f}%")

        if scheduler:
            scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_acc  = 100.0 * train_acc / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                acc1 = accuracy(logits, y, topk=(1,))[0]
                val_loss += loss.item() * x.size(0)
                val_acc  += acc1 * x.size(0) / 100.0
        val_loss /= len(test_loader.dataset)
        val_acc  = 100.0 * val_acc / len(test_loader.dataset)

        # Current LR (handles schedulers)
        if scheduler is None:
            lr_cur = optimizer.param_groups[0]["lr"]
        else:
            lr_cur = scheduler.get_last_lr()[0]

        # Log row
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, f"{train_loss:.4f}", f"{train_acc:.2f}", f"{val_loss:.4f}", f"{val_acc:.2f}", f"{lr_cur:.6f}"])

        # Save checkpoint
        ckpt = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_acc": max(best_val_acc, val_acc),
        }
        if scheduler:
            ckpt["scheduler"] = scheduler.state_dict()
        torch.save(ckpt, os.path.join(args.out, "checkpoints", f"epoch_{epoch+1}.pt"))

        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(args.out, "checkpoints", "best.pt"))

        print(f"[epoch {epoch+1}] train_loss={train_loss:.4f} train_acc={train_acc:.2f}%  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}%  best={best_val_acc:.2f}%")

    print(f"[done] best_val_acc={best_val_acc:.2f}%  logs at {log_path}")

if __name__ == "__main__":
    main()
