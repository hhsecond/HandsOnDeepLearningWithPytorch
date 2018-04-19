import torch
import numpy as np

from wavenet.exceptions import InputSizeError
import os

import torch
import torch.optim

from wavenet.networks import WaveNet as WaveNetModule


class WaveNet:
    def __init__(self, layer_size, stack_size, in_channels, res_channels, lr=0.002):

        self.net = WaveNetModule(layer_size, stack_size, in_channels, res_channels)

        self.in_channels = in_channels
        self.receptive_fields = self.net.receptive_fields

        self.lr = lr
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        self._prepare_for_gpu()

    @staticmethod
    def _loss():
        loss = torch.nn.CrossEntropyLoss()

        if torch.cuda.is_available():
            loss = loss.cuda()

        return loss

    def _optimizer(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def _prepare_for_gpu(self):
        if torch.cuda.device_count() > 1:
            print("{0} GPUs are detected.".format(torch.cuda.device_count()))
            self.net = torch.nn.DataParallel(self.net)

        if torch.cuda.is_available():
            self.net.cuda()

    def train(self, inputs, targets):
        """
        Train 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :param targets: Torch tensor [batch, timestep, channels]
        :return: float loss
        """
        outputs = self.net(inputs)

        loss = self.loss(outputs.view(-1, self.in_channels),
                         targets.long().view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def generate(self, inputs):
        """
        Generate 1 time
        :param inputs: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        outputs = self.net(inputs)

        return outputs

    @staticmethod
    def get_model_path(model_dir, step=0):
        basename = 'wavenet'

        if step:
            return os.path.join(model_dir, '{0}_{1}.pkl'.format(basename, step))
        else:
            return os.path.join(model_dir, '{0}.pkl'.format(basename))

    def load(self, model_dir, step=0):
        """
        Load pre-trained model
        :param model_dir:
        :param step:
        :return:
        """
        print("Loading model from {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        self.net.load_state_dict(torch.load(model_path))

    def save(self, model_dir, step=0):
        print("Saving model into {0}".format(model_dir))

        model_path = self.get_model_path(model_dir, step)

        torch.save(self.net.state_dict(), model_path)




class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet"""
    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=2, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=0,  # Fixed for WaveNet dilation
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(torch.nn.Module):
    """Causal Convolution for WaveNet"""
    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=2, stride=1, padding=1,
                                    bias=False)  # Fixed for WaveNet but not sure

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.data.fill_(1)

    def forward(self, x):
        output = self.conv(x)

        # remove last value for causal convolution
        return output[:, :, :-1]


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid
        # Residual network
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output += input_cut
        # Skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]
        return output, skip


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :return:
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.res_blocks = self.stack_res_block(res_channels, skip_channels)

    @staticmethod
    def _residual_block(res_channels, skip_channels, dilation):
        block = ResidualBlock(res_channels, skip_channels, dilation)

        if torch.cuda.device_count() > 1:
            block = torch.nn.DataParallel(block)

        if torch.cuda.is_available():
            block.cuda()

        return block

    def build_dilations(self):
        dilations = []
        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)
        return dilations

    def stack_res_block(self, res_channels, skip_channels):
        """
        Prepare dilated convolution blocks by layer and stack size
        :return:
        """
        res_blocks = []
        dilations = self.build_dilations()
        for dilation in dilations:
            block = self._residual_block(res_channels, skip_channels, dilation)
            res_blocks.append(block)
        return res_blocks

    def forward(self, x, skip_size):
        """
        :param x:
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []
        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)
        return torch.stack(skip_connections)


class DensNet(torch.nn.Module):
    def __init__(self, channels):
        """
        The last network of WaveNet
        :param channels: number of channels for input and output
        :return:
        """
        super(DensNet, self).__init__()

        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.softmax(output)
        return output


class WaveNetModule(torch.nn.Module):
    def __init__(self, layer_size, stack_size, in_channels, res_channels):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param in_channels: number of channels for input data. skip channel is same as input channel
        :param res_channels: number of residual channel for input, output
        :return:
        """
        super(WaveNetModule, self).__init__()
        self.receptive_fields = self.calc_receptive_fields(layer_size, stack_size)
        self.causal = CausalConv1d(in_channels, res_channels)
        self.res_stack = ResidualStack(layer_size, stack_size, res_channels, in_channels)
        self.densnet = DensNet(in_channels)

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)
        return int(num_receptive_fields)

    def calc_output_size(self, x):
        output_size = int(x.size(2)) - self.receptive_fields
        self.check_input_size(x, output_size)
        return output_size

    def check_input_size(self, x, output_size):
        if output_size < 1:
            raise InputSizeError(int(x.size(2)), self.receptive_fields, output_size)

    def forward(self, x):
        """
        The size of timestep(3rd dimention) has to be bigger than receptive fields
        :param x: Tensor[batch, timestep, channels]
        :return: Tensor[batch, timestep, channels]
        """
        output = x.transpose(1, 2)
        output_size = self.calc_output_size(output)
        output = self.causal(output)
        skip_connections = self.res_stack(output, output_size)
        output = torch.sum(skip_connections, dim=0)
        output = self.densnet(output)
        return output.transpose(1, 2).contiguous()

