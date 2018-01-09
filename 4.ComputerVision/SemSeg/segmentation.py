import torch
from torch.utils import data
from torch.autograd import Variable
import time

from torch import nn
import torch.nn.functional as F
from scipy import misc
from torchvision.models import resnet18

from dataset import CamvidDataSet
from segmentationModel import SegmentationModel

# The datafolder must be downloaed
# The path to data folder must be correct
# confusion matrix
# TODO - solve the cuda run time issue


def transfer_resnet(net):
    resnet = resnet18()

    # initial layers
    initial_params = net.init_conv.parameters()  # weights and biases for conv and bn
    conv_params = resnet.conv1.parameters()
    next(initial_params).data = next(conv_params).data
    bnparams = resnet.bn1.parameters()
    next(initial_params).data = next(bnparams).data
    next(initial_params).data = next(bnparams).data

    # encoder1 layer 1
    encoder1_param = net.encoder1.parameters()
    layer1_param = resnet.layer1.parameters()
    for _ in range(12):
        next(encoder1_param).data = next(layer1_param).data

    # encoder2 layer 2
    encoder2_param = net.encoder2.parameters()
    layer2_param = resnet.layer2.parameters()
    for _ in range(15):
        next(encoder2_param).data = next(layer2_param).data

    # encoder3 layer 3
    encoder3_param = net.encoder3.parameters()
    layer3_param = resnet.layer3.parameters()
    for _ in range(15):
        next(encoder3_param).data = next(layer3_param).data

    # encoder4 layer 4
    encoder4_param = net.encoder4.parameters()
    layer4_param = resnet.layer4.parameters()
    for _ in range(15):
        next(encoder4_param).data = next(layer4_param).data


train_encoders = False
is_cuda = torch.cuda.is_available()
if is_cuda:
    net = SegmentationModel(train_encoders=train_encoders).cuda()
else:
    net = SegmentationModel(train_encoders=train_encoders)
net.train()

if not train_encoders:
    transfer_resnet(net)

# Training

path = '/home/hhsecond/mypro/ThePyTorchBook/ThePyTorchBookDataSet/camvid'
epochs = 64
bsize = 8
dataset = CamvidDataSet('train', path)
loader = data.DataLoader(dataset, batch_size=bsize, num_workers=4, shuffle=True)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
loss_fn = nn.NLLLoss2d()


def create_image(out):
    """ Creating image from the outbatch """
    img = out[0].max(0)[1].data.cpu().numpy()
    misc.imsave('{}.png'.format(time.time()), img)


def save_model(model):
    torch.save(model.state_dict(), '{}.pth'.format(time.time()))


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
    if epoch % 50 == 0:
        net.eval()
        test_dataset = CamvidDataSet('test', path)
        test_loader = data.DataLoader(test_dataset, batch_size=bsize, num_workers=4, shuffle=True)
        loss = 0
        counter = 0
        for in_batch, target_batch in test_loader:
            if is_cuda:
                in_batch, target_batch = in_batch.cuda(), target_batch.cuda()
            out = net(Variable(in_batch))
            counter += 1
            loss += loss_fn(F.log_softmax(out, 1), Variable(target_batch)).data[0]
        loss = loss / counter
        print(' ========== Testing Loss: {:.5f} =========='.format(loss, epoch))
        create_image(out)
        save_model(net)
        net.train()
