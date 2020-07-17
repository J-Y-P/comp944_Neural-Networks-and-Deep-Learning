# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.layer1 = nn.Linear(2, num_hid)
        self.layer2 = nn.Linear(num_hid, 1)

    def forward(self, input):
        # convert to polar co-ordinates
        for i, (x, y) in enumerate(input):
            input[i] = torch.tensor([torch.sqrt(x * x + y * y), torch.atan2(y, x)])
        output = self.layer1(input)
        self.hid1 = torch.tanh(output)
        output = torch.sigmoid(self.layer2(self.hid1))
        return output

class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.layer1 = nn.Linear(2,num_hid)
        self.layer2 = nn.Linear(num_hid, num_hid)
        self.layer3 = nn.Linear(num_hid,1)

    def forward(self, input):
        output = self.layer1(input)
        self.hid1 = torch.tanh(output)
        output = self.layer2(self.hid1)
        self.hid2 = torch.tanh(output)
        output = self.layer3(self.hid2)
        output = torch.sigmoid(output)
        return output

class ShortNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(ShortNet, self).__init__()
        # INSERT CODE HERE
        self.layer1 = nn.Linear(2,num_hid)
        self.layer2 = nn.Linear(num_hid, num_hid)
        self.layer3 = nn.Linear(num_hid,1)
        self.sc1 = nn.Linear(2,num_hid)
        # self.sc2 = nn.Linear(num_hid,num_hid)

    def forward(self, input):
         # CHANGE CODE HERE
        self.hid1 = torch.tanh(self.layer1(input))
        self.hid2 = torch.tanh(self.layer2(self.hid1+self.sc1(input)))
        output = self.layer3(self.sc1(input)+self.hid1+self.hid2)
        output = torch.sigmoid((output))
        return output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        net(grid)
        if layer == 1:
            pred = (net.hid1[:,node] >= 0).float()
        elif layer == 2:
            pred = (net.hid2[:,node] >= 0).float()
        net.train()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
