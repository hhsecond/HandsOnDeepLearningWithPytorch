# copied from http://anie.me/On-Torchtext/
import os

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
    dataset = torchtext.datasets.TREC('somedummyfile', TEXT, LABEL)
    dataset.download(path)


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


# downloadTREC()
spacy_en = spacy.load('en')
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)


train, val, test = data.TabularDataset.splits(
    path='./data/', train='train.tsv',
    validation='val.tsv', test='test.tsv', format='tsv',
    fields=[('Text', TEXT), ('Label', LABEL)])
