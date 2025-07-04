import time
import torch
from utils.training_utils import get_mnist_loaders, get_cifar_loaders, train_model, count_parameters, evaluate_model
from utils.visualization_utils import plot_training_history, results_to_csv, plot_confusion_matrix, plot_gradient_flow
from utils.comparison_utils import compare_models
from models.cnn_models import SimpleCNN, CNNWithResidual, CIFARCNN
from models.fc_models import FullyConnected
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiments(device, batch_size=64, epochs=10, dataset: str = 'mnist'):
    # Load data
    if dataset == ('mnist'):
        train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

        models = {
            'FC_3layer': FullyConnected([256, 128], input_dim=28 * 28), # полносвязная модель (мало слоев)
            'SimpleCNN': SimpleCNN(input_channels=1, num_classes=10), # простая CNN
            'ResCNN': CNNWithResidual(input_channels=1, num_classes=10) # CNN с резидуальными блоками
        }
    elif dataset == 'cifar':
        train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

        models = {
            'FC_deep': FullyConnected([2048, 1024, 512, 256, 128, 64], input_dim=3 * 32 * 32), # полносвязная модель (глубокая)
            'ResCNN': CNNWithResidual(input_channels=3, num_classes=10), # CNN с резидуальными блоками
            'RegResCNN': CIFARCNN(num_classes=10) # CNN с резидуальными блоками и регуляризацией
        }
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    logger.info(f"Датасет {dataset} загружен")

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
        # Training history
        train_plot_path = f"plots/cnn_vs_fc_comparison/train_history/{dataset}/{name}_train_history.png"
        plot_training_history(history, train_plot_path)

        # Confusion matrix
        y_pred, y_true, inf_time = evaluate_model(model, test_loader, device)
        cm_path = f"plots/cnn_vs_fc_comparison/confusion_matrix/{dataset}/{name}_confusion_matrix.png"
        plot_confusion_matrix(y_true, y_pred, cm_path)

        # Gradient flow
        gf_path = f"plots/cnn_vs_fc_comparison/gradient_flow/{dataset}/{name}_gradient_flow.png"
        plot_gradient_flow(model.named_parameters(), path=gf_path, title=f"{name} Gradient Flow")

        results[name] = {
            "train_time": train_time,
            "inf_time": inf_time,
            "train_acc": history["train_accs"][-1],
            "test_acc": history["test_accs"][-1],
            "cnt_params": cnt_params
        }

    csv_path = f"results/{dataset}_comparison/{dataset}.csv"
    results_to_csv(results, csv_path)

    return histories, results


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # MNIST эксперимент
    mnist_hist, mnist_res = run_experiments(device, epochs=10, dataset='mnist')

    mnist_SimpleCNN = mnist_hist['SimpleCNN']
    mnist_ResCNN = mnist_hist['ResCNN']
    mnist_fc = mnist_hist['FC_3layer']

    compare_models(mnist_fc, mnist_SimpleCNN,
                   path=f"plots/cnn_vs_fc_comparison/compare/mnist_FC_3layer_vs_SimpleCNN.png")
    compare_models(mnist_fc, mnist_ResCNN,
                   path=f"plots/cnn_vs_fc_comparison/compare/mnist_FC_3layer_vs_ResCNN.png")

    # CIFAR эксперимент
    cifar_hist, cifar_res = run_experiments(device, epochs=10, dataset='cifar')

    cifar_RegResCNN = cifar_hist['RegResCNN']
    cifar_ResCNN = cifar_hist['ResCNN']
    cifar_fc = cifar_hist['FC_deep']

    compare_models(cifar_fc, cifar_RegResCNN,
                   path=f"plots/cnn_vs_fc_comparison/compare/cifar_FC_deep_vs_RegResCNN.png")
    compare_models(cifar_fc, cifar_ResCNN,
                   path=f"plots/cnn_vs_fc_comparison/compare/cifar_FC_deep_vs_ResCNN.png")

