import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class CustomConv2d(nn.Module):
    """
    Расширенный 2D-сверточный слой с масштабированием выходных значений.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

    def forward(self, x):
        out = self.conv(x)
        return out * self.scale

class ChannelAttention(nn.Module):
    """
    Механизм внимания по каналам (SE-блок).
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # squeeze
        y = self.avg_pool(x).view(b, c)
        # excitation
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SwishFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(x)
        grad = grad_output * (sig * (1 + x * (1 - sig)))
        return grad

class LpPool2d(Function):
    """
    Пользовательская Lp-пулинговая операция:
    forward: (sum |x|^p)^(1/p)
    backward: аналитическая производная
    """
    @staticmethod
    def forward(ctx, x, p, kernel_size, stride, padding):
        ctx.p = p
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        x_abs_p = x.abs().pow(p)
        sum_p = F.avg_pool2d(x_abs_p, kernel_size, stride, padding) * (kernel_size ** 2)
        out = sum_p.pow(1.0 / p)
        ctx.save_for_backward(x, sum_p)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, sum_p = ctx.saved_tensors
        p = ctx.p
        k2 = ctx.kernel_size ** 2
        grad_x = grad_output / (sum_p.pow((p - 1) / p) + 1e-6) * (x.abs().pow(p - 2) * x) * k2
        return grad_x


class BottleneckResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels   = out_channels
        out_expanded   = out_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_expanded, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_expanded)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_expanded:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_expanded, 1, stride, bias=False),
                nn.BatchNorm2d(out_expanded)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, width_factor=2):
        super().__init__()
        wide_channels = out_channels * width_factor

        self.conv1 = nn.Conv2d(in_channels, wide_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(wide_channels)
        self.conv2 = nn.Conv2d(wide_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CNNWithBottleneck(nn.Module):
    """
    CNN с использованием BottleneckResidualBlock
    """
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = BottleneckResidualBlock(32, 32)
        self.res2 = BottleneckResidualBlock(32*BottleneckResidualBlock.expansion, 64, stride=2)
        self.res3 = BottleneckResidualBlock(64*BottleneckResidualBlock.expansion, 64)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(64*BottleneckResidualBlock.expansion*4*4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNWithWide(nn.Module):
    """
    CNN с использованием WideResidualBlock
    """
    def __init__(self, input_channels=1, num_classes=10, width_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = WideResidualBlock(32, 32, width_factor=width_factor)
        self.res2 = WideResidualBlock(32, 64, stride=2, width_factor=width_factor)
        self.res3 = WideResidualBlock(64, 64, width_factor=width_factor)
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc = nn.Linear(64*4*4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
