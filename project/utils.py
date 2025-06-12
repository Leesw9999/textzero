import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best = float('inf')
        self.wait = 0

    def __call__(self, current_loss):
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience


def plot_history(history):
    epochs = [h['epoch'] for h in history]
    plt.plot(epochs, [h['train_loss'] for h in history], label='Train Loss')
    if 'val_loss' in history[0]:
        plt.plot(epochs, [h['val_loss'] for h in history], label='Val Loss')
    plt.plot(epochs, [h['train_acc'] for h in history], label='Train Acc')
    if 'val_acc' in history[0]:
        plt.plot(epochs, [h['val_acc'] for h in history], label='Val Acc')
    plt.xlabel('Epoch'); plt.legend(); plt.show()