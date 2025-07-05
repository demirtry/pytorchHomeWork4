import time
import torch
import logging
from utils.visualization_utils import results_to_csv, visualize_first_layer_activations, plot_feature_maps
from models.cnn_models import AdaptiveKernelCNN, AdaptiveDepthCNN, CNNWithResidual
from utils.training_utils import (
    get_cifar_loaders, get_mnist_loaders, train_model, count_parameters, evaluate_model,
    compute_receptive_field, get_model_activations)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_kernel_size_experiments(device, dataset='mnist', batch_size=64, epochs=10):

    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_channels = 1
    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_channels = 3
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    models = {
        '1x1_3x3': AdaptiveKernelCNN(kernel_size=[1, 3], input_channels=input_channels, num_classes=10, dataset=dataset),
        '3x3': AdaptiveKernelCNN(kernel_size=[3, 3], input_channels=input_channels, num_classes=10, dataset=dataset),
        '5x5': AdaptiveKernelCNN(kernel_size=[5, 5], input_channels=input_channels, num_classes=10, dataset=dataset),
        '7x7': AdaptiveKernelCNN(kernel_size=[7, 7], input_channels=input_channels, num_classes=10, dataset=dataset)
    }

    histories = {}
    results = {}

    for name, model in models.items():
        model = model.to(device)
        cnt = count_parameters(model)
        if name == '1x1_3x3':
            ks = [1, 3]
        else:
            k = int(name.split('x')[0])
            ks = [k, k]

        rf = compute_receptive_field(ks)
        logger.info(f"\nModel {name}: params={cnt}, receptive_field={rf}")

        start = time.time()
        hist = train_model(model, train_loader, test_loader, epochs=epochs, device=device)
        train_time = time.time() - start
        histories[name] = hist

        # инференс, метрики
        y_pred, y_true, inf_time = evaluate_model(model, test_loader, device)
        results[name] = {
            'params': cnt,
            'rf': rf,
            'train_time': train_time,
            'inf_time': inf_time,
            'train_acc': hist['train_accs'][-1],
            'test_acc': hist['test_accs'][-1]
        }

        # активация первого слоя
        activations = get_model_activations(model, loader=test_loader, device=device, first_layer=True)
        visualize_first_layer_activations(activations, path=f"plots/cnn_architecture_analysis/{dataset}/{name}_activations.png")

    results_to_csv(results, f"results/architecture_analysis/{dataset}_results.csv")
    return histories, results


def run_depth_experiments(device, dataset='mnist', batch_size=64, epochs=10):
    if dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        in_ch = 1
    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        in_ch = 3
    else:
        raise ValueError('Dataset not supported')

    models = {
        'CNNWithResidual': CNNWithResidual(in_ch, num_classes=10),
        '2_conv': AdaptiveDepthCNN(in_ch, dataset=dataset, num_convs=2, num_classes=10),
        '4_conv': AdaptiveDepthCNN(in_ch, dataset=dataset, num_convs=4, num_classes=10),
        '6_conv': AdaptiveDepthCNN(in_ch, dataset=dataset, num_convs=6, num_classes=10)
    }

    histories = {}
    results = {}

    for name, model in models.items():
        model = model.to(device)
        cnt = count_parameters(model)
        logger.info(f"\ndataset={dataset}, Model {name}: params={cnt}")

        plot_path = f"plots/cnn_architecture_analysis/{dataset}/grads/{name}"
        # Обучение
        start = time.time()
        hist = train_model(model, train_loader, test_loader, epochs=epochs, device=device, plot_grad=True, plot_path=plot_path)
        train_time = time.time() - start
        histories[name] = hist

        # Инференс и метрики для csv
        y_pred, y_true, inf_time = evaluate_model(model, test_loader, device)
        results[name] = {
            'params': cnt,
            'train_time': train_time,
            'inf_time': inf_time,
            'train_acc': hist['train_accs'][-1],
            'test_acc': hist['test_accs'][-1]
        }
        # У классов разная логика, поэтому разные аргументы у функции
        if name == 'CNNWithResidual':
            activations = get_model_activations(model, loader=test_loader, device=device, residual=True)
        else:
            activations = get_model_activations(model, loader=test_loader, device=device)

        feature_maps_path = f"plots/cnn_architecture_analysis/{dataset}/feature_maps/feature_map_{name}"
        plot_feature_maps(activations, feature_maps_path)

    # Сохранение результатов в csv
    results_to_csv(results, f"results/architecture_analysis/depth_analysis_{dataset}.csv")
    return histories, results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    run_kernel_size_experiments(device, dataset="mnist")
    run_kernel_size_experiments(device, dataset="cifar")

    run_depth_experiments(device, dataset="mnist")
    run_depth_experiments(device, dataset="cifar")
