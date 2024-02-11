import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.autograd import Variable
from torch.nn.utils import spectral_norm

__all__ = ['ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']


def conv3x3x3(in_planes, out_planes, stride=1):

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


class _Memory_Block(nn.Module):        
    def __init__(self, channel_dim, node_dim, class_dim, MA_rate):
        super().__init__()
        self.c = channel_dim
        self.k = node_dim
        self.a = class_dim
        self.MAR = MA_rate
        self.units = nn.Parameter(torch.randn(class_dim, node_dim, channel_dim))  
        
    def update(self, x, score, m):
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1] 
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)  
        embed_onehot_sum = embed_onehot.sum(0) 
        embed_sum = x.transpose(0, 1) @ embed_onehot  
        embed_mean = embed_sum / (embed_onehot_sum + 1e-08) 
        new_data = m * self.MAR + embed_mean.t() * (1 - self.MAR) 
        return new_data
    
    def forward(self, x, label, update=True): 
        b, c, s, h, w = x.size()      
        x = x.permute(0, 3, 4, 1, 2)  
        x = x.view(b, h*w, c*s)       
        m = self.units.data           
        mem = torch.zeros(b, self.k, self.c).cuda().data
        
        for i, (j) in enumerate(label): 
            j = j.item()
            xx = x[i,:,:]             
            mm = m[j,:,:]             
            
            xxn = F.normalize(xx, dim=1)
            mmn = F.normalize(mm, dim=1) 
            score = torch.matmul(xxn, mmn.t()) 
            
            if update: 
                mmu = self.update(xx, score, mm)
                m[j,:,:] = mmu
                mem[i,:,:] = mmu
            else:
                mem[i,:,:] = mm
                
        xn = F.normalize(x, dim=2)
        mn = F.normalize(mem, dim=2)   
        score_xm = torch.bmm(xn, mn.permute(0, 2, 1)) 
        soft_label = F.softmax(score_xm, dim=2) 
        out = torch.bmm(soft_label, mem) 
        out = out.permute(0, 2, 1).view(b, c, s, h, w)
        
        if update:
            self.units.data = m    
        memory = self.units.data
        
        return memory, out


class ResNet(nn.Module):
    def __init__(self, block, layers, sample_size=224, sample_duration=16, shortcut_type='B', num_classes=8, last_fc=True): # 8
        self.inplanes = 64
        self.last_fc = last_fc
        
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
                                  nn.BatchNorm3d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)) 
        
        self.memory_block1 = _Memory_Block(64*12, 56*56, 8, 0.999)
        self.memory_block2 = _Memory_Block(128*6, 28*28, 8, 0.999)
        self.memory_block3 = _Memory_Block(256*3, 14*14, 8, 0.999)
        
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

    def forward(self, x, label, update=True):
        
        x = self.conv(x)
        x1 = self.layer1(x)
        
        mm1, xm1 = self.memory_block1(x1, label, update)
        
        x2 = self.layer2(xm1)
        mm2, xm2 = self.memory_block2(x2, label, update)       
        
        x3 = self.layer3(xm2)
        mm3, xm3 = self.memory_block3(x3, label, update)
        
        x4 = self.layer4(xm3)
        x5 = self.ap(x4)
        x6 = x5.view(x5.size(0), -1)
        
        if self.last_fc:
            x_pre = self.fc(x6)
  
        return mm1, mm2, mm3, x_pre

class DenseBlock_Norm(nn.Module):
    def __init__(self):
        super(DenseBlock_Norm,self).__init__()
        self.first_layer = nn.Sequential(nn.Linear(256, 8))     

    def forward(self,input):
        output = self.first_layer(input)
        return output

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

net = DenseBlock_Norm()
print('DenseBlock Norm parameter count is {}'.format(count_param(net)))

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

