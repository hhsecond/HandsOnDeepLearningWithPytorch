import torch


class WaveNet(torch.nn.Module):
    def __init__(self, layer_size, stack_size, in_channels, res_channels):
        super().__init__()
        self.rf_size = sum([2 ** i for i in range(layer_size)] * stack_size)
        self.causalconv = torch.nn.Conv1d(
            in_channels, res_channels, kernel_size=2, padding=1, bias=False)
        self.res_stack = ResidualStack(
            layer_size, stack_size, res_channels, in_channels)
        self.final_conv = FinalConv(in_channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        sample_size = x.size(2)
        out_size = sample_size - self.rf_size
        if out_size < 1:
            print('Sample size has to be more than receptive field size')
        else:
            x = self.causalconv(x)[:, :, :-1]
            skip_connections = self.res_stack(x, out_size)
            x = torch.sum(skip_connections, dim=0)
            x = self.final_conv(x)
            return x.transpose(1, 2).contiguous()


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels):
        super().__init__()
        self.res_blocks = torch.nn.ModuleList()
        for s in range(stack_size):
            for l in range(layer_size):
                dilation = 2 ** l
                block = ResidualBlock(res_channels, skip_channels, dilation)
                self.res_blocks.append(block)

    def forward(self, x, skip_size):
        skip_connections = []
        for res_block in self.res_blocks:
            x, skip = res_block(x, skip_size)
            skip_connections.append(skip)
        return torch.stack(skip_connections)


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation=1):
        super().__init__()
        self.dilatedcausalconv = torch.nn.Conv1d(
            res_channels, res_channels, kernel_size=2, dilation=dilation,
            padding=0, bias=False)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)
        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        x = self.dilatedcausalconv(x)

        # PixelCNN Gate
        # ---------------------------
        gated_tanh = self.gate_tanh(x)
        gated_sigmoid = self.gate_sigmoid(x)
        gated = gated_tanh * gated_sigmoid
        # ---------------------------

        x = self.conv_res(gated)
        x += x[:, :, -x.size(2):]
        skip = self.conv_skip(gated)[:, :, -skip_size:]
        return x, skip


class FinalConv(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, 1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.softmax(x)


layer_size = 10
stack_size = 5
in_channels = 256
res_channels = 512
sample_rate = 16000
sample_size = 100000

data_dir = './test/data'
output_dir = './output'
num_steps = 100000
lr = 0.0002

net = WaveNet(layer_size, stack_size, in_channels, res_channels)
x = torch.randn([1, 10000, 256])
y = net(x)
print(y.shape)
