import sys
sys.path.insert(0, r"C:\Users\User\Downloads\textzero")

import numpy as np
from dezero.optimizers import SGD
from project.data import TextDataset, DataLoader
from project.augment import LLMDataAugmentor
from project.model import SentimentModel
from project.trainer import Trainer
from project.utils import plot_history
import project.ui as ui


if __name__ == '__main__':
    # 1) 데이터 로드
    train_ds = TextDataset('project/data/train.csv', max_len=100)
    val_ds   = TextDataset('project/data/val.csv',   max_len=100)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # 2) 데이터 증강 (소수 클래스) optional
    texts, labels = zip(*[(t,label) for t,label in train_ds.rows])
    aug = LLMDataAugmentor(api_key='YOUR_API_KEY')
    augmented = aug.augment(texts, labels, n_per_example=1)
    train_ds.rows.extend(augmented)

    # 3) 모델·옵티마이저
    model = SentimentModel(
        vocab_size=len(train_ds.vocab), embed_size=100,
        hidden_size=50, num_classes=2
    )
    optimizer = SGD(lr=0.1).setup(model)

    # 4) 학습·검증
    trainer = Trainer(model, optimizer)
    history = trainer.fit(train_loader, val_loader=val_loader, epochs=5)

    # 5) 결과 시각화 및 대시보드 실행
    plot_history(history)
    ui.run_dashboard(history, model, ["I love this movie", "Not great at all"])
    
    # main.py 끝부분

# 4) 학습·검증
trainer = Trainer(model, optimizer)
history = trainer.fit(train_loader, val_loader=val_loader, epochs=20)


# 5) 결과 저장
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history, f)

model.save_params('best_model.npz')  # best_model.npz로 저장

# 6) 시각화(기존)
plot_history(history)
# 이 줄은 남겨두셔도 되고, Streamlit UI로 대체해도 됩니다.