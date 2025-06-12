# ui.py

import sys, os
import streamlit as st
import numpy as np

# ‘dezero’ 패키지가 있는 상위 폴더를 PYTHONPATH에 추가
ROOT = r"C:\Users\User\Downloads\textzero"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dezero.core       import Variable
from dezero.optimizers import SGD
from data              import TextDataset, DataLoader, tokenize
from model             import SentimentModel
from trainer           import Trainer

st.title('Training & Dashboard')

# 데이터 로드
train_ds     = TextDataset('data/train.csv', max_len=100)
val_ds       = TextDataset('data/val.csv',   max_len=100)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

# 모델·옵티마이저
model     = SentimentModel(len(train_ds.vocab), embed_size=100, hidden_size=50, num_classes=2)
optimizer = SGD(lr=0.1).setup(model)

# 학습
st.write('학습 시작…')
trainer = Trainer(model, optimizer)
history = trainer.fit(train_loader, val_loader=val_loader, epochs=5)
st.write('학습 완료!')

# 학습/검증 곡선 시각화
st.subheader('Train/Val Loss & Accuracy')
st.line_chart({
    'Train Loss': [h['train_loss'] for h in history],
    'Val Loss':   [h['val_loss']   for h in history],
    'Train Acc':  [h['train_acc']  for h in history],
    'Val Acc':    [h['val_acc']    for h in history],
})

# 샘플 예측 (한글 레이블 매핑)
label_map = {0: '부정', 1: '긍정'}

st.subheader('Sample Predictions')
vocab = train_ds.vocab
sample_texts = ["I love you", "great at all"]
for txt in sample_texts:
    toks = tokenize(txt)
    idxs = [vocab.stoi.get(tok, vocab.stoi['<unk>']) for tok in toks]
    if len(idxs) < train_ds.max_len:
        idxs += [vocab.stoi['<pad>']] * (train_ds.max_len - len(idxs))
    else:
        idxs = idxs[:train_ds.max_len]
    x_var = Variable(np.array([idxs], dtype=np.int32))
    pred = int(np.argmax(model(x_var).data, axis=1)[0])
    st.write(f"{txt} → {label_map[pred]}")
 
