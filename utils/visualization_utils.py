import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_training_history(history, path):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def results_to_csv(results: dict, path: str):
    """
    Сохраняет результаты в CSV
    """
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv(path, sep=';', index=True, index_label='model')


def plot_confusion_matrix(y_true, y_pred, path, title: str = 'Confusion matrix'):
    """
    Строит матрицу ошибок по истинным и предсказанным меткам.
    :param y_true: истинные метки
    :param y_pred: предсказанные метки
    :param path: путь для сохранения графика
    :param title: заголовок графика
    """
    classes_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_classes = len(classes_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes_names, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes_names)

    # Текстовые аннотации
    fmt = 'd'
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def plot_gradient_flow(named_parameters, path: str, title: str = 'Gradient Flow'):
    """
    Строит и сохраняет график среднего абсолютного значения градиента по слоям.

    :param named_parameters: model.named_parameters()
    :param path: путь для сохранения графика
    :param title: заголовок графика
    """
    layers = []
    avg_grads = []

    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None and 'bias' not in name:
            layers.append(name)
            avg_grads.append(param.grad.abs().mean().item())

    if not layers:
        raise RuntimeError("Нет градиентов для отрисовки")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_grads, marker='o', linestyle='-')
    ax.hlines(0, 0, len(avg_grads)-1, colors='k', linewidth=1)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.set_ylabel('Среднее |grad|')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)