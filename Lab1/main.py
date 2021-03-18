# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class Layer():
    def __init__(self, in_features, out_features):
        self.w = np.random.normal(0, 1, size=(in_features, out_features))
        self.b = np.zeros(out_features)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.local_grad = x
        self.local_grad_b = np.ones((x.shape[0], 1))
        z = x @ self.w + self.b
        self.a = sigmoid(z)
        return self.a

    def backward(self, prev_wT_delta):
        self.up_grad = prev_wT_delta * derivative_sigmoid(self.a)
        return self.up_grad @ self.w.T

    def step(self, lr):
        grad_w = self.local_grad.T @ self.up_grad
        grad_b = self.local_grad_b.T @ self.up_grad
        self.w -= lr * grad_w
        self.b -= lr * grad_b.squeeze()


class Network():
    def __init__(self, num_features, lr=1e-3):
        self.lr = lr
        self.layers = []
        for i in range(1, len(num_features)):
            self.layers.append(Layer(num_features[i-1], num_features[i]))
        self.num_layers = len(self.layers)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i].forward(x)
        return x

    def backward(self, derivative_loss):
        prev_wT_delta = derivative_loss
        for i in reversed(range(self.num_layers)):
            prev_wT_delta = self.layers[i].backward(prev_wT_delta)

    def step(self):
        for i in range(self.num_layers):
            self.layers[i].step(self.lr)


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


def show_result(x, y, pred_y, fn=None):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes[0].set_title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            axes[0].plot(x[i,0], x[i,1], 'ro')
        else:
            axes[0].plot(x[i,0], x[i,1], 'bo')
    axes[1].set_title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            axes[1].plot(x[i,0], x[i,1], 'ro')
        else:
            axes[1].plot(x[i,0], x[i,1], 'bo')
    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn, dpi=200)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def derivative_sigmoid(x):
    return x * (1-x)


def mse_loss(pred_y, y):
    return np.mean((pred_y-y)**2)


def derivative_mse_loss(pred_y, y):
    return 2 * (pred_y-y) / y.shape[0]


def train(model, max_epochs, loss_thres, x, y):
    print('Start training')
    train_loss = []
    epoch = 1
    while True:
        pred_y = model.forward(x)
        loss = mse_loss(pred_y, y)
        model.backward(derivative_mse_loss(pred_y, y))
        model.step()
        train_loss.append(loss)
        if epoch%500 == 0:
            print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
        epoch += 1
        if loss<=loss_thres or epoch>=max_epochs:
            print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
            break
    return model, train_loss


def test(model, x, y, plot_fn):
    print('Start testing')
    pred_y = model(x)
    print('Predict:')
    print(pred_y)
    show_result(x, y, np.round(pred_y), plot_fn)


def show_learning_curve(train_loss, fn=None):
    plt.clf()
    plt.plot(np.arange(len(train_loss))+1, train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.grid()
    plt.tight_layout()
    if fn is None:
        plt.show()
    else:
        plt.savefig(fn, dpi=200)


if __name__ == '__main__':
    max_epochs = 100000
    loss_thres = 0.005
    lr = 0.5

    print('Datset: linear data')
    train_x, train_y = generate_linear(100)
    test_x, test_y = generate_linear(100)
    model_linear = Network([2, 4, 4, 1], lr=lr)
    model_linear, train_loss = train(model_linear, max_epochs, loss_thres, train_x, train_y)
    test(model_linear, test_x, test_y, 'linear_test.png')
    show_learning_curve(train_loss, 'linear_learning_curve.png')

    print('\nDataset: XOR data')
    train_x, train_y = generate_XOR_easy()
    test_x, test_y = generate_XOR_easy()
    model_xor = Network([2, 4, 4, 1], lr=0.5)
    model_xor, train_loss = train(model_xor, max_epochs, loss_thres, train_x, train_y)
    test(model_xor, test_x, test_y, 'xor_test.png')
    show_learning_curve(train_loss, 'xor_learning_curve.png')
