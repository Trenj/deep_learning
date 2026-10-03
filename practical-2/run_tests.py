"""
run_tests.py
~~~~~~~~~~~~
Прогон нескольких сетей для практики 2.
Группа: ЕТ-442
ФИО: Замятин Андрей Александрович
"""

import random
import sys
import time

import numpy as np

import mnist_loader
import network


# Список тестов: название, топология, эпохи, мини-пакет, скорость обучения,
# флаг масштабированной инициализации (только T15, см. ниже).
# T1-T9 -- исходные тесты (параметры из примеров задания и собственные:
# малая скорость, средний пакет).
# T10/T11 -- однослойная сеть [784,10] при обычных скоростях 3.0 и 1.0
# (T5 с eta=100 не годится для вывода о нужности скрытого слоя);
# T12 -- сеть [784,100,10] при средней скорости 1.0;
# T13/T14 -- промежуточные топологии [784,50,10] и [784,15,10];
# T15 -- сеть [784,100,10] при eta=3.0 с масштабированной инициализацией
# весов (вне методички, прямая проверка предположения о насыщении сигмоиды).
TESTS = [
    ("T1 базовая 784-30-10", [784, 30, 10], 30, 10, 3.0, False),
    ("T2 скрытый 100", [784, 100, 10], 30, 10, 3.0, False),
    ("T3 большой пакет", [784, 100, 10], 30, 1000, 0.001, False),
    ("T4 скорость 100", [784, 30, 10], 30, 10, 100.0, False),
    ("T5 два слоя", [784, 10], 30, 10, 100.0, False),
    ("T6 скорость 0.5", [784, 30, 10], 30, 10, 0.5, False),
    ("T7 пакет 64", [784, 30, 10], 30, 64, 3.0, False),
    ("T8 скрытый 100 скорость 0.5", [784, 100, 10], 30, 10, 0.5, False),
    ("T9 скорость 1.0", [784, 30, 10], 30, 10, 1.0, False),
    ("T10 однослойная 784-10 eta 3.0", [784, 10], 30, 10, 3.0, False),
    ("T11 однослойная 784-10 eta 1.0", [784, 10], 30, 10, 1.0, False),
    ("T12 скрытый 100 скорость 1.0", [784, 100, 10], 30, 10, 1.0, False),
    ("T13 скрытый 50 скорость 3.0", [784, 50, 10], 30, 10, 3.0, False),
    ("T14 скрытый 15 скорость 3.0", [784, 15, 10], 30, 10, 3.0, False),
    ("T15 скрытый 100 Xavier eta 3.0", [784, 100, 10], 30, 10, 3.0, True),
]


def run_one(name, sizes, epochs, batch, eta, training_data, test_data,
            seed=None, scaled_init=False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print("Seed: %d" % seed)
    print("Тест: " + name)
    print("Топология: " + str(sizes) + " Эпохи: " + str(epochs)
          + " Пакет: " + str(batch) + " Скорость: " + str(eta), flush=True)
    net = network.Network(sizes)
    if scaled_init:
        # Дополнительный эксперимент вне методички (T15):
        # масштабированная инициализация весов: w / sqrt(fan-in),
        # чтобы ослабить насыщение сигмоиды при большом fan-in.
        # network.py при этом не меняется.
        print("Инициализация: масштабированная (w / sqrt(fan-in))", flush=True)
        net.weights = [w / np.sqrt(w.shape[1]) for w in net.weights]
    t0 = time.perf_counter()
    net.SGD(training_data, epochs, batch, eta, test_data=test_data)
    dt = time.perf_counter() - t0
    print("Время обучения: %.1f с (%.2f мин)" % (dt, dt / 60))
    # Диагностика: распределение предсказаний по цифрам 0-9.
    # Если какая-то цифра не предсказана ни разу, это признак насыщения
    # сигмоиды в выходном слое (гипотеза для сетей со 100 нейронами).
    test_list = list(test_data)
    preds = [int(np.argmax(net.feedforward(x))) for x, _ in test_list]
    hist = np.bincount(preds, minlength=10)
    print("Предсказаний по цифрам: " + str(list(hist)))
    print("Итоговая точность: %d / %d" % (
        sum(int(p == y) for p, (_, y) in zip(preds, test_list)), len(test_list)))
    print("", flush=True)


def main():
    # Использование:
    #   python3 -u run_tests.py [T1 T2 ...] [--seed N] [--repeats K]
    # Без аргументов гоняем все тесты, с аргументами только перечисленные.
    # --seed задает начальное зерно ГСЧ (random + numpy) для воспроизводимости;
    # --repeats K повторяет каждый выбранный тест K раз с seed, seed+1, ...
    args = sys.argv[1:]
    seed = None
    repeats = 1
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])
            i += 2
        elif args[i] == "--repeats" and i + 1 < len(args):
            repeats = int(args[i + 1])
            i += 2
        else:
            rest.append(args[i])
            i += 1
    wanted = rest
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    for name, sizes, epochs, batch, eta, scaled in TESTS:
        if wanted and name.split()[0] not in wanted:
            continue
        for r in range(repeats):
            s = (seed + r) if seed is not None else None
            if repeats > 1:
                print("=== Повтор %d/%d ===" % (r + 1, repeats))
            run_one(name, sizes, epochs, batch, eta,
                    training_data, test_data, seed=s, scaled_init=scaled)


if __name__ == "__main__":
    main()
