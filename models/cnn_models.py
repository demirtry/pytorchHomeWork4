import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
        out = F.relu(out)
        return out


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CNNWithResidual(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.res1 = ResidualBlock(32, 32)
        self.res2 = ResidualBlock(32, 64, 2)
        self.res3 = ResidualBlock(64, 64)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AdaptiveKernelCNN(nn.Module):
    def __init__(self, kernel_size: list[int], input_channels=1, num_classes=10, dataset='mnist', combined=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        if combined:
            # Комбинированная схема: 1x1 → 3x3
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        else:
            kernel_size1, kernel_size2 = kernel_size
            padding1 = kernel_size1 // 2
            padding2 = kernel_size2 // 2

            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=kernel_size1, padding=padding1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size2, padding=padding2)

        # Вычисляем размер входа в FC слой после двух MaxPool
        if dataset == 'mnist':
            fc_input_dim = 64 * 7 * 7
        elif dataset == 'cifar':
            fc_input_dim = 64 * 8 * 8
        else:
            raise ValueError('Unknown dataset')

        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class AdaptiveDepthCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, num_convs=2, dataset='mnist'):
        super().__init__()

        self.convs = nn.ModuleList()
        channels = [input_channels] + [32, 64, 128, 256, 256, 256]  # максимум 6 conv слоёв
        for i in range(num_convs):
            in_c = channels[i]
            out_c = channels[i + 1]
            self.convs.append(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
            )

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Определим размер входа в полносвязный слой после всех сверток
        # MNIST: 28x28 → после 2 pool → 7x7
        # CIFAR: 32x32 → после 2 pool → 8x8
        if dataset == 'mnist':
            fc_input_size = channels[num_convs] * (28 // (2 ** (num_convs // 2))) ** 2
        elif dataset == 'cifar':
            fc_input_size = channels[num_convs] * (32 // (2 ** (num_convs // 2))) ** 2
        else:
            raise ValueError("Unknown dataset")

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i % 2 == 1:  # После каждого второго conv — пулинг
                x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
