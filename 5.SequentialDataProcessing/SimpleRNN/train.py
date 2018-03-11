import os
import time
from pathlib import Path
import pdb
from collections import namedtuple

import torch
from torch import optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import RNNClassifier


ConfigGen = namedtuple('ConfigGen', 'vocab_dim out_dim cells birnn dropout fc1_dim, fc2_dim')
ConfigGen.__new__.__defaults__ = (None,) * len(ConfigGen._fields)
USERHOME = str(Path.home())
batch_size = 64
inputs = data.Field(lower=True)
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, dev, test)
vector = os.path.join(USERHOME, '.vector_cache', 'glove.6B.300d.txt.pt')
pdb.set_trace()
if os.path.isfile(vector):
    # TODO - make it customizable
    inputs.vocab.vectors = torch.load(vector)
else:
    inputs.vocab.load_vectors('glove.6B.300d')
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=batch_size)

vocab_dim = len(inputs.vocab)
out_dim = len(answers.vocab)
embed_dim = 300
cells = 2
# TODO - remove bidirectional RNN for simpleRNN
birnn = True
lr = 0.01
epochs = 10
if birnn:
    cells *= 2
dropout = 0.5
fc1_dim = 50,
fc2_dim = 3
config = ConfigGen(vocab_dim, out_dim, cells, birnn, dropout, fc1_dim, fc2_dim)
model = RNNClassifier(config)
model.embed.weight.data = inputs.vocab.vectors
# TODO - convert to cuda if required

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False

for epoch in range(epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):
        model.train()
        opt.zero_grad()
        iterations += 1
        answer = model(batch)
        n_correct += (torch.max(answer, 1)
                      [1].view(batch.label.size()).data == batch.label.data).sum()
        n_total += batch.batch_size
        train_acc = 100. * n_correct / n_total
        loss = criterion(answer, batch.label)
        loss.backward()
        opt.step()
        if iterations % 5 == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                n_dev_correct += (torch.max(answer, 1)
                                  [1].view(dev_batch.label.size()) == dev_batch.label).sum()
                dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)
