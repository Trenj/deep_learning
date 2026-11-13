"""
run_training.py
~~~~~~~~~~~~~~~
Обучение сети для практики 3 (перекрестная энтропия, L2-регуляризация,
улучшенная инициализация весов) и сохранение обученной сети в файл.
Группа: ЕТ-442
ФИО: Замятин Андрей Александрович
"""

import random
import sys
import time

import numpy as np

import mnist_loader
import network2


class Tee(object):
    # Дублирует вывод print одновременно на экран и в файл results.txt.
    def __init__(self, filename):
        self.file = open(filename, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    # Использование: python3 -u run_training.py [--seed N]
    seed = None
    args = sys.argv[1:]
    if "--seed" in args:
        i = args.index("--seed")
        seed = int(args[i + 1])

    sys.stdout = Tee("results.txt")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print("Seed: %d" % seed)

    print("Загрузка данных MNIST...")
    training_data, validation_data, test_data = \
        mnist_loader.load_data_wrapper()

    print("Создание сети: network2.Network([784, 30, 10], "
          "cost=network2.CrossEntropyCost)")
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

    print("Запуск обучения: net.SGD(training_data, 30, 10, 0.5, "
          "lmbda=5.0, evaluation_data=validation_data, "
          "monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, "
          "monitor_training_accuracy=True, monitor_training_cost=True)",
          flush=True)
    t0 = time.perf_counter()
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5, lmbda=5.0,
                evaluation_data=validation_data,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                monitor_training_accuracy=True,
                monitor_training_cost=True)
    dt = time.perf_counter() - t0
    print("Время обучения: %.1f с (%.2f мин)" % (dt, dt / 60))

    # Сохранение обученной сети в файл
    net.save("trained_network.json")
    print("Обученная сеть сохранена в файл trained_network.json")

    # Проверка: загружаем сеть обратно и оцениваем на тестовой выборке
    net2_loaded = network2.load("trained_network.json")
    test_acc = net2_loaded.accuracy(test_data)
    print("Проверка загруженной из файла сети на тестовой выборке: "
          "%d / %d (%.2f%%)" % (test_acc, len(test_data),
                                100.0 * test_acc / len(test_data)))

    # Сводка прогресса обучения
    print("")
    print("Прогресс обучения (оценочная выборка, 10000 изображений):")
    print("  эпоха 0:  %d / 10000" % evaluation_accuracy[0])
    best_epoch = int(np.argmax(evaluation_accuracy))
    print("  лучший результат: %d / 10000 на эпохе %d"
          % (evaluation_accuracy[best_epoch], best_epoch))
    print("  эпоха 29: %d / 10000" % evaluation_accuracy[-1])
    print("Значение стоимостной функции на оценочной выборке:")
    print("  эпоха 0:  %.4f" % evaluation_cost[0])
    print("  эпоха 29: %.4f" % evaluation_cost[-1])
    print("Точность на обучающей выборке (50000 изображений):")
    print("  эпоха 0:  %d / 50000" % training_accuracy[0])
    print("  эпоха 29: %d / 50000" % training_accuracy[-1])


if __name__ == "__main__":
    main()
