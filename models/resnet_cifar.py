'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, feat_dim=128, low_freq=False, high_freq=False, radius=0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.normalize = NormalizeByChannelMeanStd(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


        # self.linear_contrast = nn.Linear(512*block.expansion, 128)
        dim_in = 512*block.expansion
        self.head_proj = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            # nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, 128)
        )

        self.head_pred = nn.Sequential(
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        self.low_freq = low_freq
        self.high_freq = high_freq
        self.radius = radius

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0
    def mask_radial(self, img, r):
        rows, cols = img.shape
        mask = torch.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask.cuda()
    def filter_low(self, Images, r):
        mask = self.mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
        bs, c, h, w = Images.shape
        x = Images.reshape([bs*c, h, w])
        fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
        mask = mask.unsqueeze(0).repeat([bs*c, 1, 1])
        fd = fd * mask
        fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
        fd = torch.real(fd)
        fd = fd.reshape([bs, c, h, w])
        return fd

    def filter_high(self, Images, r):
        mask = self.mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
        bs, c, h, w = Images.shape
        x = Images.reshape([bs * c, h, w])
        fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
        mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
        fd = fd * (1. - mask)
        fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
        fd = torch.real(fd)
        fd = fd.reshape([bs, c, h, w])
        return fd

        # return np.array(Images_freq_low), np.array(Images_freq_high)

    def forward(self, x, contrast=False, return_feat=False, CF=False):
        # img_org = x[0]
        # x = self.normalize(x)


        if self.low_freq:
            x = self.filter_low(x, self.radius)
            x = torch.clamp(x, 0, 1)

        if self.high_freq:
            x = self.filter_high(x, self.radius)
            x = torch.clamp(x, 0, 1)

        # img_filter = x[0]
        # import cv2
        # img_org = img_org.detach().cpu().numpy()*255.
        # img_filter = img_filter.detach().cpu().numpy()*255.
        # cv2.imwrite('org.jpg', img_org.transpose([1,2,0]))
        # cv2.imwrite('filter.jpg', img_filter.transpose([1,2,0]))
        # exit(0)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        feat = out
        if return_feat:
            return out
        if contrast:
            # out = self.linear_contrast(out)
            proj = self.head_proj(out)
            pred = self.head_pred(proj)
            proj = F.normalize(proj, dim=1)
            pred = F.normalize(pred, dim=1)
            if CF:
                return proj, pred, feat
            else:
                return proj, pred
        else:
            out = self.linear(out)
        return out


def ResNet18(num_class=10, radius=8, low_freq=False, high_freq=False):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_class, radius=radius, low_freq=low_freq, high_freq=high_freq)

def ResNet34(num_class=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_class)

def ResNet50(num_class=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_class)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()