# copied from http://anie.me/On-Torchtext/
import os

import torch
import torchtext
from torchtext import data
import spacy


def downloadTREC(path='.'):
    """
    Download doesn't actually need the fields but the init requires it,
    passing dummy fields
    """
    TEXT = torchtext.data.Field()
    LABEL = torchtext.data.Field()
    if not os.path.exists('somedummyfile'):
        f = open('somedummyfile', 'w+')
        f.close()
    # check what's the purpose of this file
    dataset = torchtext.datasets.TREC('somedummyfile', TEXT, LABEL)
    dataset.download(path)

# downloadTREC()


spacy_en = spacy.load('en')


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=True)


train, val, test = data.TabularDataset.splits(
    path='./data/', train='TRECtrain.tsv',
    validation='TRECval.tsv', test='TRECtest.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)])

TEXT.build_vocab(train, vectors="glove.6B.50d")
LABEL.build_vocab(train, vectors="glove.6B.50d")
train_iter, val_iter, test_iter = data.Iterator.splits(
    (train, val, test), sort_key=lambda x: len(x.Text),
    batch_sizes=(32, 99, 99), device=-1)


print(next(iter(test_iter)))


class DummyNN(torch.nn.Module):

    def __init__(self, emb_dim):
        self.embed = torch.nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embed.weight.data.copy_(TEXT.vocab.vectors)


# masked BPTT
# reversible tokenization
