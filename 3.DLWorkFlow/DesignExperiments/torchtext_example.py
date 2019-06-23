"""
We are using TREC data downloaded from
https://github.com/brmson/dataset-factoid-curated/blob/master/trec/ in this tutorial. But
TorchText provides convenient APIs for getting TREC dataset. We wanted to use the downloaded
files to show how to load Tabular data from the disk, in general.

For getting data directly from torchtext:

>>> import torchtext
>>> from torchtext import data
>>> import spacy
>>> spacy_en = spacy.load('en')
>>> def tokenizer(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
...
>>> TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
>>> LABEL = data.Field(sequential=False, use_vocab=True)
>>> train, test = torchtext.datasets.TREC.splits(TEXT, LABEL)
>>>
"""

import torch
from torchtext import data
import spacy


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


# The vocabulary from torchtext can be passed to an embedding layer is possible
# An example of the same is given below


class DummyNN(torch.nn.Module):

    def __init__(self, emb_dim):
        super().__init__()
        self.embed = torch.nn.Embedding(len(TEXT.vocab), emb_dim)
        self.embed.weight.data.copy_(TEXT.vocab.vectors)


net = DummyNN(50)  # 50 is inferred from the size of TEXT.vocab.vectors
print(net)
