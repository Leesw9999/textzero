import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero import Parameter, Variable

class SentimentModel:
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        self.embed_W = Parameter(
            np.random.randn(vocab_size, embed_size).astype(np.float32)
        )
        self.rnn = L.Linear(embed_size, hidden_size)
        self.fc  = L.Linear(hidden_size, num_classes)
        
    def train(self):
        # 학습 모드로 전환 (필요시 구현)
        self.training = True

    def eval(self):
        # 평가 모드로 전환 (필요시 구현)
        self.training = False

    def __call__(self, x):
        batch, seq_len = x.shape
        h = Variable(np.zeros((batch, self.rnn.out_size), np.float32))
        for t in range(seq_len):
            emb = self.embed_W.data[x.data[:, t]]
            h = F.tanh(self.rnn(Variable(emb)))
        return self.fc(h)

    def params(self):
        yield self.embed_W
        yield from self.rnn.params()
        yield from self.fc.params()