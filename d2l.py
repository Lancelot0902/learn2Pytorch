import torch
import torchvision
from torch.utils import data
from torchvision import transforms


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


def get_fashion_mnist_labels(labels):
    """ 数字标签索引及其文本名称间转换 """
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]


def get_dataloader_workers():
    """ 使用4个进程来读取数据 """
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """ 下载Fashion-MNIST数据集，然后加载到内存中 """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train,
                            batch_size,
                            shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,
                            batch_size,
                            shuffle=False,
                            num_workers=get_dataloader_workers()))


def accuracy(y_hat, y):
    """ 计算预测正确的数量 """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)    # 类似于sum中的参数keepdim
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """ 在n个变量上累加 """
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    """ 计算在指定数据集上模型的精度 """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设为评估模式
    metric = Accumulator(2)  # 创建两个变量，一个存储正确预测数量，一个存储总数量
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
