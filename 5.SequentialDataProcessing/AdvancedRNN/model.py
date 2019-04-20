import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        if config.type == 'LSTM':
            self.rnn = nn.LSTM(input_size=config.embed_dim, hidden_size=config.hidden_size,
                               num_layers=config.n_layers, dropout=config.dropout,
                               bidirectional=config.birnn)
        elif config.type == 'GRU':
            self.rnn = nn.GRU(input_size=config.embed_dim, hidden_size=config.hidden_size,
                              num_layers=config.n_layers, dropout=config.dropout,
                              bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.cells, batch_size, self.config.hidden_size
        h0 = c0 = inputs.new(*state_shape).zero_()
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        if not self.config.birnn:
            return ht[-1]
        else:
            return ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class Merger(nn.Module):

    def __init__(self, size, dropout=0.5):
        super().__init__()
        self.bn = nn.BatchNorm1d(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        prem = data[0]
        hypo = data[1]
        diff = prem - hypo
        prod = prem * hypo
        cated_data = torch.cat([prem, hypo, diff, prod], 1)
        return self.dropout(self.bn(cated_data))


class RNNClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_dim, config.embed_dim)
        self.encoder = Encoder(config)
        self.classifier = nn.Sequential(
            Merger(4 * config.hidden_size * config.n_layers, config.dropout),
            nn.Linear(
                4 * config.hidden_size * config.n_layers, config.fc1_dim),
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
        scores = self.classifier((premise, hypothesis))
        return scores
