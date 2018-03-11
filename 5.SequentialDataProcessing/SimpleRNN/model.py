import torch.nn as nn
import torch


class RNNCell(nn.Module):
    def __init__(self, vocab_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(vocab_dim + hidden_size, hidden_size)
        self.input2output = nn.Linear(vocab_dim + hidden_size, vocab_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        combined = torch.cat((inputs, hidden), 1)
        hidden = self.input2hidden(combined)
        output = self.input2output(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = RNNCell(config.vocab_dim, config.hidden_size)

    def forward(self, inputs):
        ht = self.rnn.init_hidden()
        for word in inputs.split(1, dim=1):
            outputs, ht = self.rnn(word, ht)


class Merger(nn.Module):

    def __init__(self, size, dropout=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(size * 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prem, hypo):
        return self.dropout(self.bn(torch.cat(
            [prem, hypo, prem - hypo, prem * hypo], 1)))


class RNNClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_dim, config.embed_dim)
        self.encoder = Encoder(config)
        self.merger = Merger(config.embed_dim, config.dropout)
        self.out = nn.Sequential(
            nn.Linear(4 * config.embed_dim, config.fc1_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.fc1_dim),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.fc1_dim, config.fc2_dim)
        )

    def forward(self, batch):
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        scores = self.out(self.merger(premise, hypothesis))
        return scores
