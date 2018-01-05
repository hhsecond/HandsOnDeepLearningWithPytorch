import os

import torch
from torch.utils import data
from scipy import misc
from torchvision import transforms


class CamvidDataSet(data.Dataset):
    """
    CamVid dataset Loader
    Note:
        Labels loads only pedestrian
        # TODO - do loading on gpu
    """

    def __init__(self, split, path):
        self.pedestrian_rgb = (64, 64, 0)
        inputdir_path = os.path.join(path, split)
        labledir_path = os.path.join(inputdir_path, 'labels')
        input_files = os.listdir(inputdir_path)
        try:
            input_files.remove('labels')
        except ValueError:
            raise FileNotFoundError("Couldn't find 'labels' folder in {}".format(path))
        self.files = []
        for file in input_files:
            name, ext = os.path.splitext(file)
            input_file = os.path.join(inputdir_path, file)
            label_file = os.path.join(labledir_path, '{}_L{}'.format(name, ext))
            self.files.append({'input': input_file, 'label': label_file})
        mean = [104.00699, 116.66877, 122.67892]  # found from meetshah1995/pytorch-semseg
        std = [255, 255, 255]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = misc.imread(self.files[index]['input'])
        lbl = misc.imread(self.files[index]['label'])
        return self.process(img, lbl)

    def process(self, img, lbl):
        """ converts input numpy image to torch tensor and normalize """
        # TODO - load cuda tensor if cuda_is_avialable
        img = self.transform(img)
        # TODO - check whether this conversion to long is required
        lbl = (lbl == 255).astype(int)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


if __name__ == '__main__':
    loader = CamvidDataSet(
        'train', '/home/hhsecond/mypro/ThePyTorchBook/ThePyTorchBookDataSet/camvid')
    # TODO - Do it by using Lucent
    # transforms.ToPILImage('RGB')(loader[1][1].byte()).save('image.png')
    print(loader[1][1])
