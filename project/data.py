import csv
import numpy as np
import string
from dezero import Variable

class Vocabulary:
    def __init__(self, tokens=None):
        self.stoi = {'<pad>':0, '<unk>':1}
        self.itos = {i:s for s,i in self.stoi.items()}
        if tokens:
            for t in tokens:
                self.add_token(t)
    def add_token(self, tok):
        if tok not in self.stoi:
            idx = len(self.stoi)
            self.stoi[tok] = idx
            self.itos[idx] = tok
    def __len__(self):
        return len(self.stoi)

def tokenize(text):
    # 소문자 + 구두점 제거 + 분절
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip().split()

class TextDataset:
    def __init__(self, csv_path, max_len=100, tokenizer=tokenize):
        self.rows = []
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            for text,label in reader:
                self.rows.append((text, int(label)))
        tokens = set(tok for t,_ in self.rows for tok in tokenizer(t))
        self.vocab = Vocabulary(tokens)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        text, label = self.rows[idx]
        toks = self.tokenizer(text)
        idxs = [self.vocab.stoi.get(tok,1) for tok in toks]
        if len(idxs) < self.max_len:
            idxs += [0] * (self.max_len - len(idxs))
        else:
            idxs = idxs[:self.max_len]
        return np.array(idxs, np.int32), np.int32(label)

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle: np.random.shuffle(idxs)
        for i in range(0, len(idxs), self.batch_size):
            batch = [self.dataset[j] for j in idxs[i:i+self.batch_size]]
            xs, ts = zip(*batch)
            x = Variable(np.stack(xs))
            t = Variable(np.stack(ts))
            yield x, t