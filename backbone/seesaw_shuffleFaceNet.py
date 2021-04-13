from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, ReLU6, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class h_swish(Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1, use_hs=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.unlinearity = h_swish() if use_hs else PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.unlinearity(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class PermutationBlock(Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return output

class SELayer(Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
                Linear(channel, channel // reduction),
                PReLU(channel // reduction),
                Linear(channel // reduction, channel),
                Sigmoid(),#(channel),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class seesaw_Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, use_se = 1, use_hs = 1):
        super(seesaw_Depth_Wise, self).__init__()
        self.conv_1 = Conv_block(in_c//4, out_c=groups//4, kernel=(1, 1), padding=(0, 0), stride=(1, 1), use_hs = use_hs)
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride, use_hs = use_hs)
        self.project_1 = Linear_block(groups//4, out_c//4, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_3 = Conv_block(in_c*3//4, out_c=groups*3//4, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.project_3 = Linear_block(groups*3//4, out_c*3//4, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        #self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1), groups = 2)
        self.residual = residual
        self.use_se = use_se
        self.Permu = Sequential(
            PermutationBlock(groups=2),
        )
        self.se = Sequential(
            SELayer(groups) if use_se else Sequential(),
        )
     def forward(self, x):
        if self.residual:
            short_cut = x
        x1 = x[:, :(x.shape[1]//4), :, :]
        x2 = x[:, (x.shape[1]//4):, :, :]
        x1 = self.conv_1(x1)
        x2 = self.conv_3(x2)
        x = torch.cat((x1, x2), 1)
        x = self.Permu(x)
        x = self.conv_dw(x)
        if self.use_se:
            x = self.se(x)
        x1 = x[:, :(x.shape[1]//4), :, :]
        x2 = x[:, (x.shape[1]//4):, :, :]
        x1 = self.project_1(x1)
        x2 = self.project_3(x2)
        x = torch.cat((x1, x2), 1)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class seesaw_Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_se = 1, use_hs = 1):
        super(seesaw_Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(seesaw_Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups, use_se = use_se, use_hs = use_hs))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class seesaw_shuffleFaceNet(Module):
    def __init__(self, embedding_size, out_h, out_w):
        super(seesaw_shuffleFaceNet, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = seesaw_Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, use_se = 1, use_hs = 1)
        self.conv_3 = seesaw_Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_se = 1, use_hs = 1)
        self.conv_34 = seesaw_Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, use_se = 1, use_hs = 1)
        self.conv_4 = seesaw_Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_se = 1, use_hs = 1)
        self.conv_45 = seesaw_Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, use_se = 1, use_hs = 1)
        self.conv_5 = seesaw_Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1), use_se = 1, use_hs = 1)
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(out_h,out_w), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)

        out = self.conv_45(out)

        out = self.conv_5(out)

        out = self.conv_6_sep(out)

        out = self.conv_6_dw(out)

        out = self.conv_6_flatten(out)

        out = self.linear(out)

        out = self.bn(out)
        return l2_norm(out)