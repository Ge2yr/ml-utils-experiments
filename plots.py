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
    p.add_argument('--log', default='logs/run_001.csv')
    args = p.parse_args()

    os.makedirs('figures', exist_ok=True)
    _, data = load_csv(args.log)
    epoch = data[:,0]
    train_loss = data[:,1]; val_loss = data[:,2]
    train_acc = data[:,3];   val_acc = data[:,4]

    # Loss
    plt.figure()
    plt.plot(epoch, train_loss, label='train_loss')
    plt.plot(epoch, val_loss, label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
    plt.title('Loss Curves')
    plt.savefig('figures/loss.png', dpi=180)

    # Accuracy
    plt.figure()
    plt.plot(epoch, train_acc, label='train_acc')
    plt.plot(epoch, val_acc, label='val_acc')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
    plt.title('Accuracy Curves')
    plt.savefig('figures/accuracy.png', dpi=180)

    print('[ok] wrote figures/loss.png and figures/accuracy.png')

if __name__ == '__main__':
    main()
