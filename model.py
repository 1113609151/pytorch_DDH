import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torch.nn.init as init
from config import *

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.kernel_size = _pair(kernel_size)
        self.kernel_size_1 = kernel_size
        self.stride = _pair(stride)
        self.stride_1 = stride
        
    def forward(self, x):
        output_size = (x.shape[2] - self.kernel_size_1) // self.stride_1 + 1
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, self.out_channels, self.in_channels, output_size[0], output_size[1], self.kernel_size_1**2).cuda()
        )
        if self.bias:
            self.bias = nn.Parameter(
                torch.randn(1, self.out_channels, output_size[0], output_size[1]).cuda()
            )
        else:
            self.register_parameter('bias', None)

        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        # print(self.weight.is_cuda)
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
    
class DDH(nn.Module):
    def __init__(self, hash_num, split_num, in_chaannels):
        super(DDH, self).__init__()
        self.hash_num = hash_num
        self.split_num = split_num
        self.in_chaannels = in_chaannels

        self.C1_block = nn.Sequential(
            nn.Conv2d(self.in_chaannels, 20, kernel_size=3, stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.C2_block = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=2, stride=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.C3_block = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=2, stride=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.C3_flatten = nn.Flatten()

        self.C4_block = nn.Sequential(
            #nn.Conv2dLocal from https://github.com/pytorch/pytorch/pull/1583/files
            nn.Conv2dLocal(in_channels=60, out_channels=80, in_height=3, in_width=3, kernel_size=2, stride=1, padding=0),
            # LocallyConnected2d(60, 80, kernel_size=2, stride=1, bias=True),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        self.C4_flatten = nn.Flatten()

        # self.merge = torch.cat((self.C3_flatten, self.C4_flatten), dim=1)

        # self.face_feature_layer = nn.Linear(self.merge.shape[1], self.hash_num * self.split_num)
        # init.kaiming_normal_(self.face_feature_layer.weight)

        self.C5_block = nn.Sequential(
            nn.BatchNorm1d(self.hash_num * self.split_num),
            nn.ReLU()
        )

        self.C6_block = nn.Sequential(
            nn.BatchNorm1d(self.hash_num),
            nn.Tanh()
        )

        self.liner_last = nn.Linear(self.hash_num, NB_CLASSES)

    def forward(self, x):
        x = self.C1_block(x)
        x = self.C2_block(x)
        x = self.C3_block(x)
        x_1 = self.C3_flatten(x)
        # print(x.shape)
        x = self.C4_block(x)
        x_2 = self.C4_flatten(x)

        x = torch.cat((x_1, x_2), dim=1)
        x = nn.Linear(x.shape[1], self.hash_num * self.split_num).cuda()(x)
        x = self.C5_block(x)

        outs = []
        for i in range(self.hash_num):
            slice_array = x[:, i * self.split_num : (i + 1) * self.split_num]
            fuse_layer = nn.Linear(slice_array.shape[1], 1).cuda()
            outs.append(fuse_layer(slice_array))

        x = torch.cat(outs, dim=1)
        x = self.C6_block(x)

        softmax = F.softmax(self.liner_last(x), dim=1)
        return x, softmax
    
if __name__ == "__main__":
    input_data = torch.randn(2, 3, 32, 32)
    model = DDH(10, 3, 3)
    output, softmax = model(input_data)
    print(output.shape)
    print(softmax.shape)