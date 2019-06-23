import os
import time
from pathlib import Path

import torch
from torch import optim
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import SNLIClassifier


class Config:
    """ Config class """
    ...


config = Config()
inputs = data.Field(lower=True)
answers = data.Field(sequential=False)

train, dev, test = datasets.SNLI.splits(inputs, answers)

USERHOME = str(Path.home())
vector_cache = os.path.join(USERHOME, '.vector_cache/glove.6B.300d.txt.pt')
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_layers = 1
birnn = False
lr = 0.001
epochs = 50
eval_every = 10
log_every = 4
embedding_dim = 300
projection_dim = 600
hidden_dim = 300
dropout = 0.2
tracker_dim = None
mlp_dimension = 600
predict_transitions = True
n_mlp_layers = 3

inputs.build_vocab(train, dev, test)
if os.path.isfile(vector_cache):
    inputs.vocab.vectors = torch.load(vector_cache)
else:
    inputs.vocab.load_vectors(vectors="glove.6B.300d")
    os.makedirs(os.path.dirname(vector_cache), exist_ok=True)
    torch.save(inputs.vocab.vectors, vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=batch_size, device=device)

config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = n_layers
config.d_embed = embedding_dim
config.d_proj = projection_dim
config.d_hidden = hidden_dim
config.mlp_dropout = dropout
config.embed_dropout = dropout
config.d_tracker = tracker_dim
config.birnn = birnn
config.d_mlp = mlp_dimension
config.predict = predict_transitions
config.n_mlp_layers = n_mlp_layers
if birnn:
    config.n_cells *= 2

model = SNLIClassifier(config)
model.embed.weight.data = inputs.vocab.vectors
model.to(device)

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))

print(header)

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
        if iterations % eval_every == 0:
            model.eval()
            dev_iter.init_epoch()
            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                n_dev_correct += (torch.max(
                    answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)
            print(dev_log_template.format(
                time.time() - start, epoch, iterations, 1 + batch_idx, len(train_iter),
                100. * (1 + batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0],
                train_acc, dev_acc))
        elif iterations % log_every == 0:
            print(log_template.format(
                time.time() - start, epoch, iterations, 1 + batch_idx, len(train_iter),
                100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
                n_correct / n_total * 100, ' ' * 12))
