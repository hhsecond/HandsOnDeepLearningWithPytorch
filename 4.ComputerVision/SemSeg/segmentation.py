from torch.utils import data

from dataset import CamvidDataSet
from segmentationModel import SegmentationModel

s = SegmentationModel()
# Training
path = '/home/hhsecond/mypro/ThePyTorchBook/ThePyTorchBookDataSet/camvid'
dataset = CamvidDataSet('train', path)
loader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)

for batch in loader:
    print(len(batch))
