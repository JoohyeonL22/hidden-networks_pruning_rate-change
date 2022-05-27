"""
Replications of models from Frankle et al. Lottery Ticket Hypothesis
"""
import torch
import torch.nn as nn
from utils.builder import get_builder
from args import args

import torch.autograd as autograd
from trainers.default import get_losses


global k1, k2, k3, k4, k5
k1 = 0.5
k2 = 0.5
k3 = 0.5
k4 = 0.5
k5 = 0.5


global loss_min
loss_min = 10

def getk1():
    return k1

def getk2():
    return k2

def getk3():
    return k3

def getk4():
    return k4

def getk5():
    return k5

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()

        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        #print(torch.count_nonzero(flat_out))

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class Conv2(nn.Module):
    def __init__(self):
        super(Conv2, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.linear = nn.Sequential(
            builder.conv1x1(64 * 16 * 16, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 64 * 16 * 16, 1, 1)
        out = self.linear(out)
        #print(*[(name, param.shape) for name, param in self.convs[0].named_parameters()])
        #print(self.convs[0].weight.data)
        #print(self.convs[0].scores.data)
        
        temp = float(str(get_losses())[5:10])
        if temp == 0 :
            temp = 9.499
        
        global loss_min
        #print("loss_min :", loss_min)
        #print("temp  :", temp)
        
        if loss_min > temp:
            global k1, k2, k3, k4, k5
            loss_min = temp
            #all edges
            arr1 = self.convs[0].scores.clone()
            idx1 = arr1.flatten()
            arr2 = self.convs[2].scores.clone()
            idx2 = arr2.flatten()
            arr3 = self.linear[0].scores.clone()
            idx3 = arr3.flatten()
            arr4 = self.linear[2].scores.clone()
            idx4 = arr4.flatten()
            arr5 = self.linear[4].scores.clone()
            idx5 = arr5.flatten()
            arr6 = torch.cat([idx1, idx2, idx3, idx4, idx5], dim=0)
            
            subnet = GetSubnet.apply(arr6, 0.6)
            slice1 = subnet[0:1728]
            slice2 = subnet[1728:38592]
            slice3 = subnet[38592:4232896]
            slice4 = subnet[4232896:4298432]
            slice5 = subnet[4298432:]
            tk1 = torch.count_nonzero(slice1)/1728
            tk2 = torch.count_nonzero(slice2)/36864
            tk3 = torch.count_nonzero(slice3)/4194304
            tk4 = torch.count_nonzero(slice4)/65536
            tk5 = torch.count_nonzero(slice5)/2560
            k1 = tk1.item()
            k2 = tk2.item()
            k3 = tk3.item()
            k4 = tk4.item()
            k5 = tk5.item()
            
            temp = (k1*1728 + k2*36864 + k3*4194304 + k4*65536 + k5*2560)/4300992
            print("all_k :", temp)
            #print("update_k")
        #else:
            #print("X_k")
            
        """
        print("frankle_k1 : ", k1)
        print("frankle_k2 : ", k2)
        print("frankle_k3 : ", k3)
        print("frankle_k4 : ", k4)
        print("frankle_k5 : ", k5)
        """
        
        """
        global cnt
        print("cnt1 :", cnt)
        setcnt(k1)
        print("cnt2 :", cnt)
        print("cnt3 :", getcnt())
        """
        
        """
        w1 = self.convs[0].weight.clone()
        idd1 = w1.flatten()
        w2 = self.convs[2].weight.clone()
        idd2 = w2.flatten()
        w3 = self.linear[0].weight.clone()
        idd3 = w3.flatten()
        w4 = self.linear[2].weight.clone()
        idd4 = w4.flatten()
        w5 = self.linear[4].weight.clone()
        idd5 = w5.flatten()
        w6 = torch.cat([idd1, idd2, idd3, idd4, idd5], dim=0)
        w = w6*subnet
        
        slice1 = w[0:1728]
        slice2 = w[1728:38592]
        slice3 = w[38592:4232896]
        slice4 = w[4232896:4298432]
        slice5 = w[4298432:]
        slice1 = slice1.reshape(64, 3, 3, 3)
        slice2 = slice2.reshape(64, 64, 3, 3)
        slice3 = slice3.reshape(256, 16384, 1, 1)
        slice4 = slice4.reshape(256, 256, 1, 1)
        slice5 = slice5.reshape(10, 256, 1, 1)
        self.convs[0].weight = torch.nn.Parameter(slice1)
        self.convs[2].weight = torch.nn.Parameter(slice2)
        self.linear[0].weight = torch.nn.Parameter(slice3)
        self.linear[2].weight = torch.nn.Parameter(slice4)
        self.linear[4].weight = torch.nn.Parameter(slice5)
        """
        
        """
        _, idx = arr6.flatten().sort()
        l = int((1 - 0.5) *idx.numel())
        flat_out = arr6.flatten()
        flat_out[idx[:l]] = 0
        flat_out[idx[l:]] = 1
        """
        return out.squeeze()


class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(32 * 32 * 8, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 8192, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class Conv6(nn.Module):
    def __init__(self):
        super(Conv6, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(256 * 4 * 4, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 256 * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()

class Conv8(nn.Module):
    def __init__(self):
        super(Conv8, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, 64, first_layer=True),
            nn.ReLU(),
            builder.conv3x3(64, 64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(64, 128),
            nn.ReLU(),
            builder.conv3x3(128, 128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(128, 256),
            nn.ReLU(),
            builder.conv3x3(256, 256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(256, 512),
            nn.ReLU(),
            builder.conv3x3(512, 512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(512 * 2 * 2, 256),
            nn.ReLU(),
            builder.conv1x1(256, 256),
            nn.ReLU(),
            builder.conv1x1(256, 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), 512 * 2 * 2, 1, 1)
        out = self.linear(out)
        return out.squeeze()


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        builder = get_builder()
        self.linear = nn.Sequential(
            builder.conv1x1(28 * 28, 300, first_layer=True),
            nn.ReLU(),
            builder.conv1x1(300, 100),
            nn.ReLU(),
            builder.conv1x1(100, 10),
        )

    def forward(self, x):
        out = x.view(x.size(0), 28 * 28, 1, 1)
        out = self.linear(out)
        return out.squeeze()

def scale(n):
    return int(n * args.width_mult)


class Conv4Wide(nn.Module):
    def __init__(self):
        super(Conv4Wide, self).__init__()
        builder = get_builder()

        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(128)*8*8, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(128)*8*8, 1, 1)
        out = self.linear(out)
        return out.squeeze()



class Conv6Wide(nn.Module):
    def __init__(self):
        super(Conv6Wide, self).__init__()
        builder = get_builder()
        self.convs = nn.Sequential(
            builder.conv3x3(3, scale(64), first_layer=True),
            nn.ReLU(),
            builder.conv3x3(scale(64), scale(64)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(64), scale(128)),
            nn.ReLU(),
            builder.conv3x3(scale(128), scale(128)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            builder.conv3x3(scale(128), scale(256)),
            nn.ReLU(),
            builder.conv3x3(scale(256), scale(256)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        self.linear = nn.Sequential(
            builder.conv1x1(scale(256) * 4 * 4, scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), scale(256)),
            nn.ReLU(),
            builder.conv1x1(scale(256), 10),
        )

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.size(0), scale(256) * 4 * 4, 1, 1)
        out = self.linear(out)
        return out.squeeze()