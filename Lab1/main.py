# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


def mse_loss(pred_y, y):
    return np.mean((pred_y-y)**2)


def derivative_mse_loss(pred_y, y):
    return 2 * (pred_y-y) / y.shape[0]


class Layer():
    def __init__(
            self, in_features, out_features,
            bias=True, activate=True):
        self.bias = bias
        self.activate = activate
        self.w = np.random.normal(0, 1, size=(in_features, out_features))
        if self.bias:
            self.b = np.zeros(out_features)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.local_grad = x
        self.z = x @ self.w
        if self.bias:
            self.local_grad_b = np.ones((x.shape[0], 1))
            self.z += self.b
        out = self.z
        if self.activate:
            out = sigmoid(out)
        return out

    def backward(self, prev_wT_delta):
        self.up_grad = prev_wT_delta
        if self.activate:
            self.up_grad *= derivative_sigmoid(self.z)
        return self.up_grad @ self.w.T

    def step(self, lr):
        grad_w = self.local_grad.T @ self.up_grad
        self.w -= lr * grad_w
        if self.bias:
            grad_b = self.local_grad_b.T @ self.up_grad
            self.b -= lr * grad_b.squeeze()


class Network():
    def __init__(
            self, num_features, lr=1e-3,
            no_bias=False, no_activate=False):
        self.lr = lr
        self.bias = not no_bias
        self.activate = not no_activate

        self.layers = []
        for i in range(1, len(num_features)):
            self.layers.append(Layer(
                num_features[i-1], num_features[i],
                bias=self.bias, activate=self.activate))
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


def train(model, x, y, batch_size=-1):
    print('------------------')
    print('| Start training |')
    print('------------------')
    max_epochs = 100000
    loss_thres = 0.005
    bs = batch_size if batch_size!=-1 else x.shape[0]
    train_loss = []
    epoch = 1
    while True:
        start_idx = 0
        end_idx = min(start_idx+bs, x.shape[0])
        batch_num = 0
        loss = 0
        while True:
            bx = x[start_idx:end_idx]
            by = y[start_idx:end_idx]
            pred_y = model.forward(bx)
            loss += mse_loss(pred_y, by)
            model.backward(derivative_mse_loss(pred_y, by))
            model.step()
            batch_num += 1

            start_idx = end_idx
            end_idx = min(start_idx+bs, x.shape[0])
            if start_idx >= x.shape[0]:
                break

        loss /= batch_num
        train_loss.append(loss)
        if epoch%500 == 0:
            print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
        epoch += 1
        if loss<=loss_thres or epoch>=max_epochs:
            print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
            break

    return model, train_loss


# def train(model, x, y):
#     print('------------------')
#     print('| Start training |')
#     print('------------------')
#     max_epochs = 100000
#     loss_thres = 0.005
#     train_loss = []
#     epoch = 1
#     while True:        
#         pred_y = model.forward(x)
#         loss = mse_loss(pred_y, y)
#         model.backward(derivative_mse_loss(pred_y, y))
#         model.step()
#         train_loss.append(loss)
#         if epoch%500 == 0:
#             print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
#         epoch += 1
#         if loss<=loss_thres or epoch>=max_epochs:
#             print(f'[Epoch:{epoch:6}] [Loss: {loss:.6f}]')
#             break
#     return model, train_loss


def test(model, x, y, plot_fn):
    print('-----------------')
    print('| Start testing |')
    print('-----------------')
    pred_y = model(x)
    loss = mse_loss(pred_y, y)
    pred_y_rounded = np.round(pred_y)
    pred_y_rounded[pred_y_rounded<0] = 0
    correct = np.sum(pred_y_rounded==y)
    acc = 100 * correct / len(y)
    show_result(x, y, pred_y_rounded, acc, plot_fn)
    print('Prediction:\n', pred_y)
    print(f'Testing loss: {loss:.3f}')
    print(f'Acc: {correct}/{len(y)} ({acc:.1f}%)')


def show_learning_curve(train_loss, fn=None):
    plt.clf()
    plt.plot(np.arange(len(train_loss))+1, train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    if fn is None:
        plt.show()
    else:
        plt.savefig(os.path.join('plots', fn), dpi=200)
    np.save(os.path.join('train_loss', fn.replace('.png', '.npy')), np.array(train_loss))


def show_data(x, y, title, fn=None):
    plt.clf()
    color = ['bo', 'ro']
    for i in range(x.shape[0]):
        plt.plot(x[i,0], x[i,1], color[int(y[i]==0)])
    plt.title(title)
    plt.axis('square')
    if fn is None:
        plt.show()
    else:
        plt.savefig(os.path.join('plots', fn), dpi=200)


def show_result(x, y, pred_y, acc, fn=None):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    axes[0].set_title('Ground truth', fontsize=18)
    axes[1].set_title('Predict result', fontsize=18)
    color = ['bo', 'ro']
    for i in range(x.shape[0]):
        axes[0].plot(x[i,0], x[i,1], color[int(y[i]==0)])
        axes[1].plot(x[i,0], x[i,1], color[int(pred_y[i]==0)])
        if pred_y[i] != y[i]:
            circle = plt.Circle((x[i,0], x[i,1]), 0.06, color='black', fill=False)
            axes[1].add_patch(circle)
    axes[0].set_aspect('equal', 'box')
    axes[1].set_aspect('equal', 'box')
    plt.suptitle(f'Acc: {acc:.1f}%', fontsize=16)
    if fn is None:
        plt.show()
    else:
        plt.savefig(os.path.join('plots', fn), dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--num_features', type=str, default='2-4-4-1')
    parser.add_argument('--no_bias', action='store_true', default=False)
    parser.add_argument('--no_activate', action='store_true', default=False)
    opts = parser.parse_args()
    print(opts)

    batch_size = opts.batch_size
    lr = opts.lr
    num_features = opts.num_features.split('-')
    num_features = [int(n) for n in num_features]
    no_bias = opts.no_bias
    no_activate = opts.no_activate

    exp_name = f'bs{batch_size}_lr{lr}_{opts.num_features}'
    if no_bias:
        exp_name += '_no_bias'
    if no_activate:
        exp_name += '_no_activate'

    os.makedirs('train_loss', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    print('Datset: linear data')
    train_x, train_y = generate_linear(100)
    show_data(train_x, train_y, title='Linear', fn='linear.png')
    test_x, test_y = generate_linear(100)
    model_linear = Network(num_features, lr=lr, no_bias=no_bias, no_activate=no_activate)
    model_linear, train_loss = train(model_linear, train_x, train_y, batch_size)
    test(model_linear, test_x, test_y, f'linear_{exp_name}_test.png')
    show_learning_curve(train_loss, f'linear_{exp_name}_lc.png')

    print()

    print('Dataset: XOR data')
    train_x, train_y = generate_XOR_easy()
    show_data(train_x, train_y, title='XOR', fn='xor.png')
    test_x, test_y = generate_XOR_easy()
    model_xor = Network(num_features, lr=lr, no_bias=no_bias, no_activate=no_activate)
    model_xor, train_loss = train(model_xor, train_x, train_y, batch_size)
    test(model_xor, test_x, test_y, f'xor_{exp_name}_test.png')
    show_learning_curve(train_loss, f'xor_{exp_name}_lc.png')

    # max_epochs = 100000
    # loss_thres = 0.005
    # os.makedirs('train_loss', exist_ok=True)
    # os.makedirs('plots', exist_ok=True)
    # train_x, train_y = generate_linear(100)
    # show_data(train_x, train_y, title='Linear', fn='linear.png')
    # test_x, test_y = generate_linear(100)
    # for lr in [0.05, 0.1, 0.5, 1]:
    #     for num_features in ['2-2-2-1', '2-4-4-1', '2-8-8-1']:
    #         exp_name = f'lr{lr}_{num_features}'
    #         num_features = num_features.split('-')
    #         num_features = [int(n) for n in num_features]
    #         print('Datset: linear data')
    #         model_linear = Network(num_features, lr=lr)
    #         model_linear, train_loss = train(model_linear, max_epochs, loss_thres, train_x, train_y)
    #         test(model_linear, test_x, test_y, f'linear_{exp_name}_test.png')
    #         show_learning_curve(train_loss, f'linear_{exp_name}_lc.png')
    # print()
    # train_x, train_y = generate_XOR_easy()
    # show_data(train_x, train_y, title='XOR', fn='xor.png')
    # test_x, test_y = generate_XOR_easy()
    # for lr in [0.05, 0.1, 0.5, 1]:
    #     for num_features in ['2-2-2-1', '2-4-4-1', '2-8-8-1']:
    #         exp_name = f'lr{lr}_{num_features}'
    #         num_features = num_features.split('-')
    #         num_features = [int(n) for n in num_features]
    #         print('Dataset: XOR data')
    #         model_xor = Network(num_features, lr=lr)
    #         model_xor, train_loss = train(model_xor, max_epochs, loss_thres, train_x, train_y)
    #         test(model_xor, test_x, test_y, f'xor_{exp_name}_test.png')
    #         show_learning_curve(train_loss, f'xor_{exp_name}_lc.png')
