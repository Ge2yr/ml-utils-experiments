import argparse, os, csv, random
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--model', default='resnet18')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--opt', choices=['sgd','adam'], default='sgd')
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=3)
    args = p.parse_args()

    os.makedirs('logs', exist_ok=True)

    # Simulated run log with plausible curves (demo & plotting)
    random.seed(args.seed); np.random.seed(args.seed)
    rows = [('epoch','train_loss','val_loss','train_acc','val_acc')]
    train_loss, val_loss = 1.8, 2.0
    train_acc,  val_acc  = 0.40, 0.35
    for epoch in range(1, args.epochs+1):
        train_loss *= (0.85 + 0.02*np.random.rand())
        val_loss   *= (0.88 + 0.03*np.random.rand())
        train_acc   = min(0.99, train_acc + 0.05 + 0.02*np.random.rand())
        val_acc     = min(0.99,  val_acc + 0.04 + 0.02*np.random.rand())
        rows.append((epoch, round(train_loss,4), round(val_loss,4),
                     round(train_acc,4), round(val_acc,4)))

    out = 'logs/run_001.csv'
    with open(out,'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f'[ok] wrote {out}')

if __name__ == '__main__':
    main()
