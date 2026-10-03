"""
check_init.py
~~~~~~~~~~~~~
Проверка насыщения сигмоиды при инициализации randn (без обучения).
Группа: ЕТ-442
ФИО: Замятин Андрей Александрович

Считает модуль предактиваций |z| выходного слоя и распределение
предсказаний до обучения на первых 2000 примерах тестовой выборки.
Предположение: при инициализации np.random.randn (std=1) и fan-in 784
значения |z| велики, сигмоида насыщена, предсказания схлопываются.
"""

import random

import numpy as np

import mnist_loader
import network

N_EXAMPLES = 2000
SEED = 0


def stats_for(sizes):
    random.seed(SEED)
    np.random.seed(SEED)
    net = network.Network(sizes)
    _, _, test_data = mnist_loader.load_data_wrapper()
    sample = list(test_data)[:N_EXAMPLES]
    z_abs = []
    preds = []
    for x, _ in sample:
        a = x
        for b, w in zip(net.biases, net.weights):
            z = np.dot(w, a) + b
            a = 1.0 / (1.0 + np.exp(-z))
        z_abs.extend(np.abs(z).ravel().tolist())
        preds.append(int(np.argmax(a)))
    z_abs = np.array(z_abs)
    hist = np.bincount(preds, minlength=10)
    print("Топология: %s (seed %d, примеров %d)" % (sizes, SEED, N_EXAMPLES))
    print("|z| выходного слоя: среднее %.2f, максимум %.2f" % (float(z_abs.mean()), float(z_abs.max())))
    print("Предсказаний по цифрам до обучения: %s" % (list(hist),))


def main():
    print("Проверка инициализации (без обучения)")
    for sizes in ([784, 30, 10], [784, 100, 10]):
        stats_for(sizes)


if __name__ == "__main__":
    main()
