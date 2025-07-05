import time
import torch
from models.custom_layers import CustomConv2d, ChannelAttention, SwishFunction, LpPool2d
from utils.training_utils import get_mnist_loaders, get_cifar_loaders, train_model, count_parameters
from utils.visualization_utils import plot_training_history, results_to_csv
from models.custom_layers import CNNWithBottleneck, CNNWithWide
from models.cnn_models import CNNWithResidual
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def layers_test():
    # Тест CustomConv2d
    layer = CustomConv2d(3, 8, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 16, 16)
    y = layer(x)
    assert y.shape == (2, 8, 16, 16), logger.error(f"CustomConv2d shape {y.shape}, expected (2,8,16,16)")
    logger.info("CustomConv2d тест прошел")

    # Тест ChannelAttention
    att = ChannelAttention(8)
    y_att = att(y)
    assert y_att.shape == (2, 8, 16, 16), logger.error(f"ChannelAttention shape {y_att.shape}, expected (2,8,16,16)")
    logger.info("ChannelAttention тест прошел")

    # Тест SwishFunction
    x_sw = torch.randn(1, 8, 4, 4, dtype=torch.double, requires_grad=True)
    y_sw = SwishFunction.apply(x_sw)
    assert y_sw.shape == x_sw.shape, logger.error(f"SwishFunction shape {y_sw.shape}, expected {x_sw.shape}")
    logger.info("SwishFunction тест прошел")

    # Тест LpPool2d
    x_lp = torch.randn(1, 8, 6, 6, dtype=torch.double, requires_grad=True)
    p, ks, st, pd = 3.0, 2, 2, 0
    y_lp = LpPool2d.apply(x_lp, p, ks, st, pd)
    expected_shape = (1, 8, (6 - ks + 2*pd)//st + 1, (6 - ks + 2*pd)//st + 1)
    assert y_lp.shape == expected_shape, logger.error(f"LpPool2d shape {y_lp.shape}, expected {expected_shape}")
    logger.info("LpPool2d тест прошел")


def run_residual_blocks_experiments(device, batch_size=64, epochs=10, dataset: str = 'mnist'):

    if dataset == ('mnist'):
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)
        input_channels = 1
    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)
        input_channels = 3
    else:
        raise ValueError(f"Dataset {dataset} not supported")


    models = {
        'CNNWithBottleneck': CNNWithBottleneck(input_channels=input_channels),
        'CNNWithWide': CNNWithWide(input_channels=input_channels),
        'CNNWithResidual': CNNWithResidual(input_channels=input_channels)
    }

    histories = {}
    results = {}

    for name, model in models.items():
        model = model.to(device)
        cnt_params = count_parameters(model)

        logger.info(f"\n=== Training {name} for {dataset} | params: {cnt_params} ===")

        start_train = time.time()
        history = train_model(model, train_loader, test_loader,
                              epochs=epochs, device=device)
        train_time = time.time() - start_train

        histories[name] = history
        train_plot_path = f"plots/custom_layers_experiments/{dataset}/{name}_train_history.png"
        plot_training_history(history, train_plot_path)


        results[name] = {
            "train_time": train_time,
            "train_acc": history["train_accs"][-1],
            "test_acc": history["test_accs"][-1],
            "cnt_params": cnt_params
        }

    csv_path = f"results/custom_layers_experiments/{dataset}_results.csv"
    results_to_csv(results, csv_path)

    return histories, results


if __name__ == '__main__':
    layers_test()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    run_residual_blocks_experiments(device=device, dataset="mnist")
    run_residual_blocks_experiments(device=device, dataset="cifar")
