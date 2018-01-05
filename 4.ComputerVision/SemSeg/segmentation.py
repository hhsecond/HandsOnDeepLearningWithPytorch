import torch
from torch.utils import data
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from dataset import CamvidDataSet
from segmentationModel import SegmentationModel

# The datafolder must be downloaed
# The path to data folder must be correct
# confusion matrix
# TODO - solve the cuda run time issue

is_cuda = torch.cuda.is_available()
if is_cuda:
    net = SegmentationModel().cuda()
else:
    net = SegmentationModel()
net.train()
# Training
path = '/home/hhsecond/mypro/ThePyTorchBook/ThePyTorchBookDataSet/camvid'
epochs = 64
bsize = 2
dataset = CamvidDataSet('train', path)
loader = data.DataLoader(dataset, batch_size=bsize, num_workers=4, shuffle=True)
optimizer = torch.optim.Adam(net.parameters())
loss_fn = nn.NLLLoss2d()

for epoch in range(epochs):
    for in_batch, target_batch in loader:
        if is_cuda:
            in_batch, target_batch = in_batch.cuda(), target_batch.cuda()
        optimizer.zero_grad()
        out = net(Variable(in_batch))
        loss = loss_fn(F.log_softmax(out, 1), Variable(target_batch))
        loss.backward()
        optimizer.step()
        # TODO - make the visualization for this soon
    print('Training Loss: {:.5f}, Epochs: {:3d}'.format(loss.data[0], epoch))
    if epoch % 5 == 0:
        net.eval()
        test_dataset = CamvidDataSet('test', path)
        test_loader = data.DataLoader(dataset, batch_size=bsize, num_workers=4, shuffle=True)
        loss = 0
        counter = 0
        for in_batch, target_batch in test_loader:
            if is_cuda:
                in_batch, target_batch = in_batch.cuda(), target_batch.cuda()
            out = net(Variable(in_batch))
            counter += 1
            loss += loss_fn(F.log_softmax(out, 1), Variable(target_batch))
        loss = loss.data[0] / counter
        print(' ========== Testing Loss: {:.5f} =========='.format(loss, epoch))
        net.train()
