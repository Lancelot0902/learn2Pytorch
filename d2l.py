import torch


def synthetic_data(w, b, num_examples):
    """ 生成线性回归的模型 """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def linreg(X, w, b):
    """ 线性回归模型 """
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """ 均方损失 """
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):
    """ 小批量随机梯度下降 """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
