import torch.nn as nn
from collections import OrderedDict


class Flatten(nn.Module):
    """
    Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
    """
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        n, m = input.shape[0], input.shape[1]
        return input.view(n, m)


def all_cnn_module():
    """
    Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Use a AvgPool2d to pool and then your Flatten layer as your final layers.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """

    Sequential = nn.Sequential(OrderedDict([
          ('drop1', nn.Dropout(0.2)),
          ('conv1', nn.Conv2d(3, 96, 3, 1, 1, bias=False)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(96, 96, 3, 1, 1, bias=False)),
          ('relu2', nn.ReLU()),
          ('conv3', nn.Conv2d(96, 96, 3, 2, 1, bias=False)),
          ('relu3', nn.ReLU()),
          ('drop2', nn.Dropout(0.5)),
          ('conv4', nn.Conv2d(96, 192, 3, 1, 1, bias=False)),
          ('relu4', nn.ReLU()),
          ('conv5', nn.Conv2d(192, 192, 3, 1, 1, bias=False)),
          ('relu5', nn.ReLU()),
          ('conv6', nn.Conv2d(192, 192, 3, 2, 1, bias=False)),
          ('relu6', nn.ReLU()),
          ('drop3', nn.Dropout(0.5)),
          ('conv7', nn.Conv2d(192, 192, 3, 1, 0, bias=False)),
          ('relu7', nn.ReLU()),
          ('conv8', nn.Conv2d(192, 192, 1, 1, 0, bias=False)),
          ('relu8', nn.ReLU()),
          ('conv9', nn.Conv2d(192, 10, 1, 1, 0, bias=False)),
          ('relu9', nn.ReLU()),
          ('avg', nn.AvgPool2d(6)),
          ('flatten', Flatten())
        ]))
    
    return Sequential