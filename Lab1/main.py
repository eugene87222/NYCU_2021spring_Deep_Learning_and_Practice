# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i,0], x[i,1], 'ro')
        else:
            plt.plot(x[i,0], x[i,1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i,0], x[i,1], 'ro')
        else:
            plt.plot(x[i,0], x[i,1], 'bo')

    plt.show()


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def derivative_sigmoid(x):
    return x * (1-x)


def loss(y, pred_y):
    return np.mean((y-pred_y))


def derivative_loss(y, pred_y):
    return 2 * (y-pred_y) / y.shape[0]


class Layer():
    def __init(self, in_features, out_features):
        self.w = np.random.normal(0, 1, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def forward(self, x):
        self.local_grad = x
        z = x @ self.w + self.b
        self.a = sigmoid(z)
        return self.a

    def backward(self, prev_wT_delta):
        self.up_grad = prev_wT_delta * derivative_sigmoid(self.a)
        return self.w.T @ self.up_grad

    def step(self, lr):
        grad_w = self.local_grad.T @ self.up_grad
        grad_b = self.up_grad
        self.w -= lr * grad_w
        self.b -= lr * grad_b
        return grad_w, grad_b


class Network():
    def __init__(self, num_features, lr=1e-3):
        self.lr = lr
        self.layers = []
        for i in range(1, len(num_features)):
            self.layers.append(Layer(num_features[i-1], num_features[i]))
        self.num_layers = len(self.layers)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i].forward(x)
        return x

    def backward(self, derivative_loss):
        prev_wT_delta = derivative_loss
        for i in reversed(range(self.num_layers)):
            prev_wT_delta = self.layers[i].backward(prev_wT_delta)

    def step(self):
        grads_w, grads_b = [], []
        for i in range(self.num_layers):
            grad_w, grad_b = self.layers[i].step(self.learning_rate)
            grads_w.append(grad_w)
            grads_b.append(grad_b)
        return grads_w, grads_b


if __name__ == '__main__':
    x, y = generate_linear(100)
    x, y = generate_XOR_easy()
