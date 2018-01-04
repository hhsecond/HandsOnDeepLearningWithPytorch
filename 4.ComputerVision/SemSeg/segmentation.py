import torch
from torch.utils import data
from torch.autograd import Variable

from loss import CrossEntropyLoss2d
from dataset import CamvidDataSet
from segmentationModel import SegmentationModel

# The datafolder must be downloaed
# The path must to data folder must be correct
# confusion matrix

net = SegmentationModel()
# Training
path = '/home/hhsecond/mypro/ThePyTorchBook/ThePyTorchBookDataSet/camvid'
dataset = CamvidDataSet('train', path)
loader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
optimizer = torch.optim.Adam(net.parameters())
loss_fn = CrossEntropyLoss2d()


for in_batch, out_batch in loader:
    out = net(Variable(in_batch))
    print(out.size(), out_batch.size())
    print(type(out_batch.max()))
    if out_batch.max() == 0:
        print('not doing it >>>>>>>>>>>>')
    else:
        print(out_batch.max())
        loss = loss_fn(out, Variable(out_batch))
