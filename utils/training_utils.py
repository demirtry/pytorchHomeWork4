import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import logging
import time
from utils.visualization_utils import plot_gradient_flow


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    """
    Запускает эпоху
    :param model: модель
    :param data_loader: загрузчик датасета
    :param criterion: критерий потерь
    :param optimizer: оптимизатор
    :param device: устройство выполнения cpu или cuda
    :param is_test: флаг тест
    :return:
    """
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu', plot_grad: bool=False, plot_path: str=None):
    """
    Запускает обучение
    :param model: модель
    :param train_loader: тренировочный загрузчик
    :param test_loader: тестовый загрузчик
    :param epochs: число эпох
    :param lr: learning rate
    :param device: устройство выполнения cpu или cuda
    :return:
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        logger.info('-' * 50)

        # делаю график градиентов для слоев при необходимости
        if plot_grad and (epoch + 1) % 2 == 0:
            plot_gradient_flow(model.named_parameters(), path=f"{plot_path}_Epoch_{epoch + 1}.png")

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device):
    """
    Оценивает модель: возвращает предсказания, исходные метки и время инференса
    :param model: обученная модель
    :param dataloader: даталоадер
    :param device: девайс cuda или cpu
    :return:
    """
    model.eval()
    all_preds = []
    all_targets = []
    start_time = time.time()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_targets.append(y.cpu())

    inf_time = time.time() - start_time
    return torch.cat(all_preds), torch.cat(all_targets), inf_time


def compute_receptive_field(kernel_sizes: list):
    """
    Вычисляет рецептивное поле для последовательности conv слоев с одинаковыми параметрами.
    :param kernel_sizes: список размеров ядер
    """
    strides = [1] * len(kernel_sizes)
    rf = 1
    jump = 1
    for k, s in zip(kernel_sizes, strides):
        rf = rf + (k - 1) * jump
        jump *= s
    return rf


def get_batch(model, loader, device, num_images=4):
    """Возвращает первые num изображений из даталоадера"""
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs[:num_images].to(device)

    return imgs


def get_first_layer_model_activations(model, batch):
    """Сохраняет активации первого conv-слоя для первых num изображений"""
    with torch.no_grad():
        activations = F.relu(model.conv1(batch))  # [B, C, H, W]
    activations = activations.cpu().numpy()

    return activations


def get_model_activations(model, loader, device, num_images=4, first_layer=False, residual=False):
    """Сохраняет активации первого conv-слоя для первых num изображений"""

    batch = get_batch(model, loader, device, num_images)

    if first_layer:
        return get_first_layer_model_activations(model, batch)

    activations = []
    x = batch
    if residual:
        x = model.conv1(x)
        x = model.bn1(x)
        x = F.relu(x)
        activations.append(x.detach().cpu())
        for res_block in (model.res1, model.res2, model.res3):
            x = res_block(x)
            activations.append(x.detach().cpu())
        return activations

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            x = layer(x)
            x = torch.relu(x)
            activations.append(x.detach().cpu())
        elif isinstance(layer, torch.nn.MaxPool2d) or isinstance(layer, torch.nn.AdaptiveAvgPool2d):
            x = layer(x)

    return activations


class MNISTDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class CIFARDataset(Dataset):
    def __init__(self, train=True, transform=None):
        super().__init__()
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNISTDataset(train=True, transform=transform)
    test_dataset = MNISTDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_cifar_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = CIFARDataset(train=True, transform=transform)
    test_dataset = CIFARDataset(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader