# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.lin = nn.Sequential( 
            nn.Linear(28*28, 10),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = x.view(-1,28*28)
        return self.lin(x) # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.full =  nn.Sequential( 
            nn.Linear(28 * 28, 700),
            nn.Tanh(),
            nn.Linear(700, 10))

    def forward(self, x):
        x = x.view(-1,28*28)
        return F.log_softmax(self.full(x), dim = 1) # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.layer1 = nn.Sequential( 
            nn.Conv2d(1, 25, kernel_size=3),    # 20 * 26 * 26
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 20 * 13 * 13

        self.layer2 = nn.Sequential( 
            nn.Conv2d(25, 50, kernel_size=3),   # 50 * 11 * 11
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 50 * 5 * 5


        self.fc = nn.Sequential(
            nn.Linear(50 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x # CHANGE CODE HERE




