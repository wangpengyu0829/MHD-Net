import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=3, last_fc=True): # 8
        self.last_fc = last_fc
        self.inplanes = 64
        
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
                                  nn.BatchNorm3d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)) 
        
        self.conv2d1 = nn.Conv2d(768, 192, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2d2 = nn.Conv2d(768, 192, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(768, 192, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.conv3d1 = nn.Conv3d(256, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3d2 = nn.Conv3d(320, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3d3 = nn.Conv3d(448, 256, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        
        last_duration = math.ceil(sample_duration/16)
        last_size = math.ceil(sample_size/32)
        
        self.ap = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512*block.expansion*2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes*block.expansion, stride=stride)
            else:
                downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                                           nn.BatchNorm3d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, mm1, mm2, mm3, smpre):
        
        x = self.conv(x)
        smpreu = smpre.unsqueeze(2).unsqueeze(3)  
        
        x1 = self.layer1(x)
        b, c1, s1, h1, w1 = x1.shape
        pmm1 = mm1.permute(0, 2, 1)      
        
        rmm1 = pmm1.unsqueeze(0).repeat(b,1,1,1) 
        
        premm1 = torch.sum(smpreu*rmm1, dim=1).view(b, c1*s1, h1, w1) 
        premm1 = self.conv2d1(premm1)     
        xmm1 = torch.cat((x1, (premm1.unsqueeze(2)).repeat(1,1,s1,1,1)), dim=1) 
        xmm1 = self.conv3d1(xmm1)         
        
        x2 = self.layer2(xmm1+x1)
        b, c2, s2, h2, w2 = x2.shape
        pmm2 = mm2.permute(0, 2, 1)       
        rmm2 = pmm2.unsqueeze(0).repeat(b,1,1,1) 
        premm2 = torch.sum(smpreu*rmm2, dim=1).view(b, c2*s2, h2, w2) 
        premm2 = self.conv2d2(premm2)    
        xmm2 = torch.cat((x2, (premm2.unsqueeze(2)).repeat(1,1,s2,1,1)), dim=1) 
        xmm2 = self.conv3d2(xmm2)         
        
        x3 = self.layer3(xmm2+x2)
        b, c3, s3, h3, w3 = x3.shape
        pmm3 = mm3.permute(0, 2, 1)       
        rmm3 = pmm3.unsqueeze(0).repeat(b,1,1,1) 
        premm3 = torch.sum(smpreu*rmm3, dim=1).view(b, c3*s3, h3, w3) 
        premm3 = self.conv2d3(premm3)     
        xmm3 = torch.cat((x3, (premm3.unsqueeze(2)).repeat(1,1,s3,1,1)), dim=1) 
        xmm3 = self.conv3d3(xmm3)         
        
        x4 = self.layer4(xmm3+x3)
        
        x5 = self.ap(x4)
        x6 = x5.view(x5.size(0), -1)
        
        if self.last_fc:
            x_pre = self.fc(x6)
        
        return xmm1+x1, xmm2+x2, xmm3+x3, x_pre, F.sigmoid(x6)


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()
    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(ft_begin_index))
    ft_module_names.append('fc')
    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})
    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model"""
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model

def resnet18(**kwargs):
    """Constructs a ResNet-18 model"""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    """Constructs a ResNet-34 model"""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    """Constructs a ResNet-50 model"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    """Constructs a ResNet-101 model"""
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def resnet200(**kwargs):
    """Constructs a ResNet-101 model"""
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model




