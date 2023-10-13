import torch as t
from torch import nn
from torch.nn import Module

# the residual component
class Residual(Module):

    def __init__(self, ch, kernel_size=3, padding=1):
        super(Residual, self).__init__()
        self.convs = t.nn.ModuleList([nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=padding) for i in range(2)])
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.convs[0](x))
        x_ = self.convs[1](x)
        return self.act(x + x_)


# we call it image net
class ImgNet(nn.Module):
    def __init__(self, class_num=10):
        super(ImgNet, self).__init__()
        self.seq = t.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5),
            Residual(12),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3),
            Residual(12),
            nn.AvgPool2d(kernel_size=3)
        )
        
        self.fc = t.nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, class_num)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.seq(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        if self.training:
            return x
        else:
            return self.softmax(x)
