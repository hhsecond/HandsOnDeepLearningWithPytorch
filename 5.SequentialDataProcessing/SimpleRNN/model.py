import torch.nn as nn
import torch


class RNNCell(nn.Module):
    def __init__(self, embed_dim, hidden_size, vocab_dim):
        super().__init__()

        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(embed_dim + hidden_size, hidden_size)
        self.input2output = nn.Linear(embed_dim + hidden_size, vocab_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        combined = torch.cat((inputs, hidden), 2)
        hidden = self.input2hidden(combined)
        # Since it's encoder
        # We are not concerened about output
        # output = self.input2output(combined)
        # output = self.softmax(output)
        return hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class Encoder(nn.Module):

    def __init__(self, embed_dim, vocab_dim, hidden_size):
        super(Encoder, self).__init__()
        self.rnn = RNNCell(embed_dim, hidden_size, vocab_dim)

    def forward(self, inputs):
        # .size(1) dimension is batch size
        ht = self.rnn.init_hidden(inputs.size(1))
        for word in inputs.split(1, dim=0):
            ht = self.rnn(word, ht)
        return ht


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
        cated_data = torch.cat([prem, hypo, diff, prod], 2)
        cated_data = cated_data.squeeze()
        return self.dropout(self.bn(cated_data))


class RNNClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_dim, config.embed_dim)
        self.encoder = Encoder(
            config.embed_dim, config.vocab_dim, config.hidden_size)
        self.classifier = nn.Sequential(
            Merger(4 * config.hidden_size, config.dropout),
            nn.Linear(4 * config.hidden_size, config.fc1_dim),
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
