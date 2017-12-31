import torch
from torch import nn


class SegmentationModel(nn.Module):
    """
    LinkNet for Semantic segmentation. Inspired heavily by
    https://github.com/meetshah1995/pytorch-semseg
    """

    def __init__(self, n_classes=21):
        super().__init__()

