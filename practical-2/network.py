""""""""""""""""""""""""""""""""""""""" 
network.py 
Модуль создания и обучения нейронной сети для распознавания рукописных цифр 
с использованием метода градиентного спуска.  
Группа: ЕТ-442
ФИО: Замятин Андрей Александрович 
"""""""""""""""""""""""""""""""""""""""

#### Библиотеки
# Стандартные библиотеки
import random  # библиотека функций для генерации случайных значений

# Сторонние библиотеки
import numpy as np  # библиотека функций для работы с матрицами

""" ---Раздел описаний--- """

def sigmoid(z):  # определение сигмоидальной функции активации
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):  # Производная сигмоидальной функции
    return sigmoid(z) * (1 - sigmoid(z))


""" --Описание класса Network--"""
class Network(object):  # используется для описания нейронной сети

    def __init__(self, sizes):  # конструктор класса
        # self – указатель на объект класса
        # sizes – список размеров слоев нейронной сети
        self.num_layers = len(sizes)  # задаем количество слоев нейронной сети
        self.sizes = sizes  # задаем список размеров слоев нейронной сети
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # случайные начальные смещения
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]  # случайные начальные веса связей

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(  # Стохастический градиентный спуск
        self,             # указатель на объект класса
        training_data,    # обучающая выборка
        epochs,           # количество эпох обучения
        mini_batch_size,  # размер подвыборки
        eta,              # скорость обучения
        test_data         # тестирующая выборка
    ):
        test_data = list(test_data)  # список объектов тестирующей выборки
        n_test = len(test_data)      # длина тестирующей выборки
        training_data = list(training_data)  # список объектов обучающей выборки
        n = len(training_data)       # размер обучающей выборки

        for j in range(epochs):  # цикл по эпохам
            random.shuffle(training_data)  # перемешиваем элементы обучающей выборки
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]  # создаем подвыборки
            for mini_batch in mini_batches:  # цикл по подвыборкам
                self.update_mini_batch(mini_batch, eta)  # шаг градиентного спуска
            print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))

    def update_mini_batch(  # Шаг градиентного спуска
        self,        # указатель на объект класса
        mini_batch,  # подвыборка
        eta          # скорость обучения
    ):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # градиенты dC/db (нули)
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # градиенты dC/dw (нули)

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)  # градиенты для (x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]  # обновляем веса
        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]  # обновляем смещения

    def backprop(  # Алгоритм обратного распространения
        self,  # указатель на объект класса
        x,     # вектор входных сигналов
        y      # ожидаемый вектор выходных сигналов
    ):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # определение переменных
        activation = x
        activations = [x]  # список выходных сигналов по всем слоям
        zs = []  # список активационных потенциалов по всем слоям

        # прямое распространение
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # обратное распространение
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  # BP1
        nabla_b[-1] = delta  # BP3
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # BP4

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp  # BP2
            nabla_b[-l] = delta  # BP3
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())  # BP4

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):  # Оценка прогресса в обучении
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):  # частные производные стоимостной функции
        return (output_activations - y)
""" --Конец описания класса Network--"""
""" --- Конец раздела описаний--- """
""" ---Тело программы (демо из методички, только при прямом запуске)--- """
if __name__ == "__main__":
    net = Network([2, 3, 1])  # создаем нейронную сеть из трех слоев
    """ ---Конец тела программы--- """
    """ Вывод результата на экран: """
    print('Сеть net:')
    print('Количество слоев:', net.num_layers)
    for i in range(net.num_layers):
        print('Количество нейронов в слое', i, ':', net.sizes[i])
    for i in range(net.num_layers-1):
        print('W_', i+1, ':')
        print(np.round(net.weights[i], 2))
        print('b_', i+1, ':')
        print(np.round(net.biases[i], 2))
