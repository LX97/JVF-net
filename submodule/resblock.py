import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Block(nn.Module):  # ResNet-18, 34, omiting norm layers
    def __init__(self, in_channels, out_channels, hidden_channels=None, k_size=3, pad=1, activation=F.relu, downsample=False):
        super(Block, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.activation = activation
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, k_size, padding=pad)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, k_size, padding=pad)
        self.learnable_sc = (in_channels != out_channels) or downsample
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv_sc(x)
            if self.downsample:
                return F.avg_pool2d(x, 2)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, k_size, padding=pad)
        self.conv2 = nn.Conv2d(out_channels, out_channels, k_size, padding=pad)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def residual(self, x):
        h = x
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        return h

    def shortcut(self, x):
        return self.conv_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


if __name__ == '__main__':
    block1 = OptimizedBlock(3, 64)
    print(block1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    block1 = block1.to(device)
    summary(block1, (3,224,224))