import torch
import torch.nn as nn
class basicblock(nn.Module):
    def __init__(self,in_channel,out_channel):
        #super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out
