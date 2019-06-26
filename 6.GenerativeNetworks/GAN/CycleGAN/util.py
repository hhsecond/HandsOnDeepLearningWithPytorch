from argparse import ArgumentParser
import torch


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--dataroot', type=str, default='./datasets/maps/',
        help=('path to images; can be downloaded using '
              'https://github.com/junyanz/CycleGAN/blob/master/datasets/download_dataset.sh'))
    parser.add_argument(
        '--input_nc', type=int, default=3,
        help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        '--output_nc', type=int, default=3,
        help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--n_cpu', default=0, type=int, help='# threads for loading data')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument(
        "--decay_epoch", type=int, default=100,
        help="epoch from which to start lr decay")
    parser.add_argument(
        "--cuda", type=bool, default=torch.cuda.is_available(),
        help='CUDA availability check')
    parser.add_argument('--size', type=int, default=256, help='crop to this size')
    args = parser.parse_args(args=[])
    return args
