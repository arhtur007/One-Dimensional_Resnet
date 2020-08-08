import torch
import torch.nn as nn
import math

from torch.autograd import Function


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs):

        energy = self.projection(encoder_outputs)
        weights = self.softmax(energy.squeeze(-1))
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)

        return outputs, weights


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class AttentiveStatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(AttentiveStatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel
        self.attention = SelfAttention(2048)

    def forward(self, x):
        x = x.transpose(2, 1)
        means, w = self.attention(x)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1

        residuals = (x * w.unsqueeze(-1)) - means.unsqueeze(1)

        numerator = torch.sum(residuals**2, dim=1)

        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=-1)
        return x


def conv1x3(in_channels, out_channels, stride=1, padding=0, bias=False):
    "3x3 convolution with padding"
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=bias)

def conv1x1(in_channels, out_channels, stride=1, padding=0, bias=False):
    "3x3 convolution with padding"
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=padding, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.relu = ReLU(inplace=True)

        self.conv1 = conv1x1(in_channels=in_channels, out_channels=out_channels, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv1x3(in_channels=out_channels, out_channels=out_channels, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = conv1x1(in_channels=out_channels, out_channels=out_channels*self.expansion, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                conv1x1(in_channels=in_channels, out_channels=out_channels*self.expansion, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels*self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet50(nn.Module):

    def __init__(self, layers=[2,2,2,2], expansion=4, input_size=39, output_size=64):

        super(ResNet50, self).__init__()

        self.relu = ReLU(inplace=True)
        self.expansion = expansion

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, bias=False)

        self.layer1 = self.make_layer(in_channels=64, out_channels=64, layer=layers[0], stride=1)
        self.layer2 = self.make_layer(in_channels=256, out_channels=128, layer=layers[1], stride=2)
        self.layer3 = self.make_layer(in_channels=512, out_channels=256, layer=layers[2], stride=2)
        self.layer4 = self.make_layer(in_channels=1024, out_channels=512, layer=layers[3], stride=2)

        self.pooling = AttentiveStatsPool()
        self.fc1 = nn.Linear(1024*self.expansion, 512)
        self.fc2 = nn.Linear(512, output_size)
        
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, out_channels, layer, stride):

        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsampling=True))
        for i in range(1, layer):
            layers.append(Bottleneck(out_channels*self.expansion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.pooling(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)
 
        return x
