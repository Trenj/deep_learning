"""
run_tests.py
~~~~~~~~~~~~
Прогон нескольких сетей для практики 2.
Группа: ЕТ-442
ФИО: Замятин Андрей Александрович
"""

import sys

import mnist_loader
import network


# Список тестов: название, топология, эпохи, мини-пакет, скорость обучения
TESTS = [
    ("T1 базовая 784-30-10", [784, 30, 10], 30, 10, 3.0),
    ("T2 скрытый 100", [784, 100, 10], 30, 10, 3.0),
    ("T3 большой пакет", [784, 100, 10], 30, 1000, 0.001),
    ("T4 скорость 100", [784, 30, 10], 30, 10, 100.0),
    ("T5 два слоя", [784, 10], 30, 10, 100.0),
    ("T6 скорость 0.5", [784, 30, 10], 30, 10, 0.5),
    ("T7 пакет 64", [784, 30, 10], 30, 64, 3.0),
    ("T8 скрытый 100 скорость 0.5", [784, 100, 10], 30, 10, 0.5),
    ("T9 скорость 1.0", [784, 30, 10], 30, 10, 1.0),
]


def main():
    # Без аргументов гоняем все тесты, с аргументами только перечисленные
    wanted = sys.argv[1:]
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    for name, sizes, epochs, batch, eta in TESTS:
        if wanted and name.split()[0] not in wanted:
            continue
        print("Тест: " + name)
        print("Топология: " + str(sizes) + " Эпохи: " + str(epochs) + " Пакет: " + str(batch) + " Скорость: " + str(eta))
        net = network.Network(sizes)
        net.SGD(training_data, epochs, batch, eta, test_data=test_data)
        print("")


if __name__ == "__main__":
    main()
