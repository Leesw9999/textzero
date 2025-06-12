import time
import numpy as np
import dezero.functions as F

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def fit(self, train_loader, val_loader=None, epochs=5):
        history = []
        for epoch in range(1, epochs+1):
            # Training
            sum_loss, sum_acc, cnt = 0.0, 0, 0
            for x, t in train_loader:
                y = self.model(x)
                loss = F.softmax_cross_entropy(y, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.update()
                sum_loss += float(loss.data)
                pred = np.argmax(y.data, axis=1)
                sum_acc += np.sum(pred == t.data)
                cnt     += len(t.data)
            logs = {'epoch':epoch, 'train_loss':sum_loss/cnt, 'train_acc':sum_acc/cnt}

            # Validation
            if val_loader is not None:
                self.model.eval()
                v_loss, v_acc, v_cnt = 0.0, 0, 0
                for x_val, t_val in val_loader:
                    yv = self.model(x_val)
                    lv = F.softmax_cross_entropy(yv, t_val)
                    v_loss += float(lv.data)
                    pv = np.argmax(yv.data, axis=1)
                    v_acc  += np.sum(pv == t_val.data)
                    v_cnt  += len(t_val.data)
                logs['val_loss'] = v_loss/v_cnt
                logs['val_acc']  = v_acc/v_cnt
                self.model.train()

            print(logs)
            history.append(logs)
        return history