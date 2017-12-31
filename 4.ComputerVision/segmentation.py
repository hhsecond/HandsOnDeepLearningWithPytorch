import torch
from torch.utils import data

path = '/home/hhsecond/mypro/camvid'


class CamvidLoader(data.Dataset):
    """
    CamVid dataset Loader
    dataset downloaded from https://github.com/mostafaizz/camvid
    Note:
        Labels loads only pedestrian
    """

    def __init__(self, path):
        pass
