import torch
from torch.utils import data
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from dataset import CamvidDataSet
from segmentationModel import SegmentationModel

# The datafolder must be downloaed
# The path must to data folder must be correct
# confusion matrix

is_cuda = torch.cuda.is_available()
if is_cuda:
    net = SegmentationModel().cuda()
else:
    net = SegmentationModel()
# Training
path = '/home/hhsecond/mypro/ThePyTorchBook/ThePyTorchBookDataSet/camvid'
dataset = CamvidDataSet('train', path)
loader = data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
optimizer = torch.optim.Adam(net.parameters())
loss_fn = nn.NLLLoss2d()


for in_batch, target_batch in loader:
    if is_cuda:
        in_batch, target_batch = in_batch.cuda(), target_batch.cuda()
    optimizer.zero_grad()
    out = net(Variable(in_batch))
    loss = loss_fn(F.log_softmax(out, 1), Variable(target_batch))
    loss.backward()
    optimizer.step()
    # TODO - make the visualization for this soon
    print('Loss: {:.5f}'.format(loss.data[0]))
