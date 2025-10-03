import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    data = np.array(rows[1:], dtype=float)
    return hdr, data

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log', default='runs/cifar10_resnet18/log.csv')
    args = p.parse_args()

    hdr, data = load_csv(args.log)
    epoch = data[:,0]
    train_loss = data[:,1]; train_acc = data[:,2]
    val_loss   = data[:,3]; val_acc   = data[:,4]
    lr         = data[:,5]

    os.makedirs('figures', exist_ok=True)

    plt.figure()
    plt.plot(epoch, train_loss, label='train_loss')
    plt.plot(epoch, val_loss,   label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
    plt.title('Loss'); plt.savefig('figures/loss.png', dpi=180)

    plt.figure()
    plt.plot(epoch, train_acc, label='train_acc')
    plt.plot(epoch, val_acc,   label='val_acc')
    plt.xlabel('epoch'); plt.ylabel('accuracy (%)'); plt.legend()
    plt.title('Accuracy'); plt.savefig('figures/accuracy.png', dpi=180)

    plt.figure()
    plt.plot(epoch, lr, label='lr')
    plt.xlabel('epoch'); plt.ylabel('learning rate'); plt.legend()
    plt.title('LR schedule'); plt.savefig('figures/lr.png', dpi=180)

    print('[ok] wrote figures/loss.png, figures/accuracy.png, figures/lr.png')

if __name__ == '__main__':
    main()
