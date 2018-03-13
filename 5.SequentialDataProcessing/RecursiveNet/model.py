import itertools

import torch
import torch.nn as nn


def tree_lstm(c1, c2, lstm_in):
    a, i, f1, f2, o = lstm_in.chunk(5, 1)
    c = a.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
    h = o.sigmoid() * c.tanh()
    return h, c


def bundle(lstm_iter):
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return torch.cat(lstm_iter, 0).chunk(2, 1)


def unbundle(state):
    if state is None:
        return itertools.repeat(None)
    return torch.split(torch.cat(state, 1), 1, 0)


class Reduce(nn.Module):

    def __init__(self, size, tracker_size=None):
        super().__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def forward(self, left_in, right_in, tracking=None):
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left[0])
        lstm_in += self.right(right[0])
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking[0])
        out = unbundle(tree_lstm(left[1], right[1], lstm_in))
        return out


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, predict):
        super().__init__()
        self.rnn = nn.LSTMCell(3 * size, tracker_size)
        if predict:
            self.transition = nn.Linear(tracker_size, 4)
        self.state_size = tracker_size

    def reset_state(self):
        self.state = None

    def forward(self, bufs, stacks):
        buf = bundle(buf[-1] for buf in bufs)[0]
        stack1 = bundle(stack[-1] for stack in stacks)[0]
        stack2 = bundle(stack[-2] for stack in stacks)[0]
        x = torch.cat((buf, stack1, stack2), 1)
        if self.state is None:
            self.state = 2 * [x.data.new(x.size(0), self.state_size).zero_()]
        self.state = self.rnn(x, self.state)
        if hasattr(self, 'transition'):
            return unbundle(self.state), self.transition(self.state[0])
        return unbundle(self.state), None


class SPINN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.d_hidden == config.d_proj / 2
        self.reduce = Reduce(config.d_hidden, config.d_tracker)
        if config.d_tracker is not None:
            self.tracker = Tracker(config.d_hidden, config.d_tracker,
                                   predict=config.predict)

    def forward(self, buffers, transitions):
        buffers = [list(torch.split(b.squeeze(1), 1, 0))
                   for b in torch.split(buffers, 1, 1)]
        stacks = [[buf[0], buf[0]] for buf in buffers]

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        else:
            assert transitions is not None

        if transitions is not None:
            num_transitions = transitions.size(0)
        else:
            num_transitions = len(buffers[0]) * 2 - 3

        for i in range(num_transitions):
            if transitions is not None:
                trans = transitions[i]
            if hasattr(self, 'tracker'):
                tracker_states, trans_hyp = self.tracker(buffers, stacks)
                if trans_hyp is not None:
                    trans = trans_hyp.max(1)[1]
                    # if transitions is not None:
                    #     trans_loss += F.cross_entropy(trans_hyp, trans)
                    #     trans_acc += (trans_preds.data == trans.data).mean()
                    # else:
                    #     trans = trans_preds
            else:
                tracker_states = itertools.repeat(None)
            lefts, rights, trackings = [], [], []
            batch = zip(trans.data, buffers, stacks, tracker_states)
            for transition, buf, stack, tracking in batch:
                if transition == 3:  # shift
                    stack.append(buf.pop())
                elif transition == 2:  # reduce
                    rights.append(stack.pop())
                    lefts.append(stack.pop())
                    trackings.append(tracking)
            if rights:
                reduced = iter(self.reduce(lefts, rights, trackings))
                for transition, stack in zip(trans.data, stacks):
                    if transition == 2:
                        stack.append(next(reduced))
        # if trans_loss is not 0:
        return bundle([stack.pop() for stack in stacks])[0]


class Feature(nn.Module):

    def __init__(self, size, dropout):
        super().__init__()
        self.bn = nn.BatchNorm1d(size * 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prem, hypo):
        return self.dropout(self.bn(torch.cat(
            [prem, hypo, prem - hypo, prem * hypo], 1)))


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=config.rnn_dropout,
                           bidirectional=config.birnn)

    def forward(self, inputs, _):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = inputs.data.new(*state_shape).zero_()
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        if not self.config.birnn:
            return ht[-1]
        else:
            return ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)


class SNLIClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = nn.Linear(config.d_embed, config.d_proj)
        self.embed_bn = nn.BatchNorm1d(config.d_proj)
        self.embed_dropout = nn.Dropout(p=config.embed_dropout)
        self.encoder = SPINN(config) if config.spinn else Encoder(config)
        feat_in_size = config.d_hidden * (
            2 if self.config.birnn and not self.config.spinn else 1)
        self.feature = Feature(feat_in_size, config.mlp_dropout)
        self.mlp_dropout = nn.Dropout(p=config.mlp_dropout)
        self.relu = nn.ReLU()
        mlp_in_size = 4 * feat_in_size
        mlp = [nn.Linear(mlp_in_size, config.d_mlp), self.relu,
               nn.BatchNorm1d(config.d_mlp), self.mlp_dropout]
        for i in range(config.n_mlp_layers - 1):
            mlp.extend([nn.Linear(config.d_mlp, config.d_mlp), self.relu,
                        nn.BatchNorm1d(config.d_mlp), self.mlp_dropout])
        mlp.append(nn.Linear(config.d_mlp, config.d_out))
        self.out = nn.Sequential(*mlp)

    def forward(self, batch):
        prem_embed = self.embed(batch.premise)
        hypo_embed = self.embed(batch.hypothesis)
        prem_embed = self.embed_dropout(self.embed_bn(prem_embed))
        hypo_embed = self.embed_dropout(self.embed_bn(hypo_embed))
        if hasattr(batch, 'premise_transitions'):
            prem_trans = batch.premise_transitions
            hypo_trans = batch.hypothesis_transitions
        else:
            prem_trans = hypo_trans = None
        premise = self.encoder(prem_embed, prem_trans)
        hypothesis = self.encoder(hypo_embed, hypo_trans)
        scores = self.out(self.feature(premise, hypothesis))
        return scores
