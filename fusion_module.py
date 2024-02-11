import torch
import torch.nn as nn
from involution_cuda_att import involution


def conv1x1x1(in_planes, out_planes, stride=1):
    # 1x1x1 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5x5(in_planes, out_planes, stride=1):
    # 5x5x5 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

def conv7x7x7(in_planes, out_planes, stride=1):
    # 7x7x7 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


class Fusion_Block1(nn.Module):
    def __init__(self, inplanes, stride=1, downsample=None, rr=4, ch=12, k=7):
        super(Fusion_Block1, self).__init__()  
        self.rr = rr
        self.APool_1 = nn.AdaptiveAvgPool3d((ch, k, k))
        self.conv1_CT = conv1x1x1(inplanes, inplanes)
        self.conv1_WSI = conv1x1x1(inplanes, inplanes)
        
        self.conv1_Att = conv5x5x5(k*k*ch, 1+ch)
        self.sigmoid = nn.Sigmoid()
        self.conv1_Inv = conv1x1x1(k*k*ch, k*k)

        self.conv3_rr = conv3x3x3(inplanes, inplanes//rr)
        self.Inv = involution(kernel_size=k, stride=1, groups=ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(inplanes+ch*(inplanes//rr))
        self.bn2 = nn.BatchNorm3d(inplanes+ch*(inplanes//rr))
        self.proj1 = conv1x1x1(inplanes+ch*(inplanes//rr), inplanes+ch*(inplanes//rr))
        self.proj2 = conv1x1x1(inplanes+ch*(inplanes//rr), inplanes)
        
    def forward(self, x_CT, x_WSI):
        b, c, t, h, w = x_CT.size()
        
        x_CT1 = self.conv1_CT(x_CT)      
        x_WSI1 = self.conv1_WSI(x_WSI)   
        x_WSI1 = self.APool_1(x_WSI1)    
        
        b, c, s, k, k = x_WSI1.size()
        x_CT2 = x_CT1.view(b, c, t*h*w).permute(0,2,1)
        x_WSI2 = x_WSI1.view(b, c, s*k*k)
        multi = torch.bmm(x_CT2, x_WSI2)  
        feature = multi.view(b, t, h, w, s*k*k).permute(0,4,1,2,3)
               
        Att = self.sigmoid(self.conv1_Att(feature))    
        
        x_WSI3 = self.conv3_rr(x_WSI) 
        b, cr, s, h, w = x_WSI3.size()
        x_WSI4 = x_WSI3.view(b, cr*s, h, w)
        x_WSI5 = x_WSI4.unsqueeze(2).repeat(1,1,t,1,1)
        x_FS = torch.cat((x_CT, x_WSI5), 1)
        
        Att_CT = Att[:,0:1,:,:,:] 
        Att_CT = Att_CT.repeat(1,c,1,1,1) 
        
        Att_WSI = Att[:,1:,:,:,:].view(b,s,t,h*w).repeat(1,1,c//self.rr,1)  
        Att_WSI = Att_WSI.view(b, s*(c//self.rr), t, h, w)
        Att_FS  = torch.cat((Att_CT, Att_WSI), 1)
        
        x_Att = x_FS*Att_FS + x_FS


        Inv = self.conv1_Inv(feature) 
        Inv = Inv.permute(0,2,1,3,4).view(b, t, k, k, h, w) 
        x_Att1 = self.relu1(self.bn1(self.proj1(x_Att)))
        out = self.Inv(x_Att1, Inv).view(b, c+s*(c//self.rr), t, h, w)  
        out = self.proj2(self.bn2(out)+x_Att)                     
        return out 


class Fusion_Block2(nn.Module):
    def __init__(self, inplanes, stride=1, downsample=None, rr=4, ch=6, k=5):
        super(Fusion_Block2, self).__init__()
        self.rr = rr
        self.APool_1 = nn.AdaptiveAvgPool3d((ch, k, k))
        self.conv1_CT = conv1x1x1(inplanes, inplanes)
        self.conv1_WSI = conv1x1x1(inplanes, inplanes)
        
        self.conv1_Att = conv5x5x5(k*k*ch, 1+ch)
        self.sigmoid = nn.Sigmoid()
        self.conv1_Inv = conv1x1x1(k*k*ch, k*k)

        self.conv3_rr = conv3x3x3(inplanes, inplanes//rr)
        self.Inv = involution(kernel_size=k, stride=1, groups=ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(inplanes+ch*(inplanes//rr))
        self.bn2 = nn.BatchNorm3d(inplanes+ch*(inplanes//rr))
        self.proj1 = conv1x1x1(inplanes+ch*(inplanes//rr), inplanes+ch*(inplanes//rr))
        self.proj2 = conv1x1x1(inplanes+ch*(inplanes//rr), inplanes)

    def forward(self, x_CT, x_WSI):
        b, c, t, h, w = x_CT.size()
        
        x_CT1 = self.conv1_CT(x_CT)      
        x_WSI1 = self.conv1_WSI(x_WSI)   
        x_WSI1 = self.APool_1(x_WSI1)    
        
        b, c, s, k, k = x_WSI1.size()
        x_CT2 = x_CT1.view(b, c, t*h*w).permute(0,2,1)
        x_WSI2 = x_WSI1.view(b, c, s*k*k)
        multi = torch.bmm(x_CT2, x_WSI2)  
        feature = multi.view(b, t, h, w, s*k*k).permute(0,4,1,2,3)
               
        Att = self.sigmoid(self.conv1_Att(feature))    
        
        x_WSI3 = self.conv3_rr(x_WSI) 
        b, cr, s, h, w = x_WSI3.size()
        x_WSI4 = x_WSI3.view(b, cr*s, h, w)
        x_WSI5 = x_WSI4.unsqueeze(2).repeat(1,1,t,1,1)
        x_FS = torch.cat((x_CT, x_WSI5), 1)
        
        Att_CT = Att[:,0:1,:,:,:] 
        Att_CT = Att_CT.repeat(1,c,1,1,1) 
        
        Att_WSI = Att[:,1:,:,:,:].view(b,s,t,h*w).repeat(1,1,c//self.rr,1) 
        Att_WSI = Att_WSI.view(b, s*(c//self.rr), t, h, w)
        Att_FS  = torch.cat((Att_CT, Att_WSI), 1)
        
        x_Att = x_FS*Att_FS + x_FS

        Inv = self.conv1_Inv(feature) 
        Inv = Inv.permute(0,2,1,3,4).view(b, t, k, k, h, w) 
        x_Att1 = self.relu1(self.bn1(self.proj1(x_Att)))
        out = self.Inv(x_Att1, Inv).view(b, c+s*(c//self.rr), t, h, w)  
        out = self.proj2(self.bn2(out)+x_Att)                 
        return out 


class Fusion_Block3(nn.Module):
    def __init__(self, inplanes, stride=1, downsample=None, rr=4, ch=3, k=3):
        super(Fusion_Block3, self).__init__()  
        self.rr = rr
        self.APool_1 = nn.AdaptiveAvgPool3d((ch, k, k))
        self.conv1_CT = conv1x1x1(inplanes, inplanes)
        self.conv1_WSI = conv1x1x1(inplanes, inplanes)
        
        self.conv1_Att = conv5x5x5(k*k*ch, 1+ch)
        self.sigmoid = nn.Sigmoid()
        self.conv1_Inv = conv1x1x1(k*k*ch, k*k)

        self.conv3_rr = conv3x3x3(inplanes, inplanes//rr)
        self.Inv = involution(kernel_size=k, stride=1, groups=ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(inplanes+ch*(inplanes//rr))
        self.bn2 = nn.BatchNorm3d(inplanes+ch*(inplanes//rr))
        self.proj1 = conv1x1x1(inplanes+ch*(inplanes//rr), inplanes+ch*(inplanes//rr))
        self.proj2 = conv1x1x1(inplanes+ch*(inplanes//rr), inplanes)

    def forward(self, x_CT, x_WSI):
        b, c, t, h, w = x_CT.size()
        
        x_CT1 = self.conv1_CT(x_CT)      
        x_WSI1 = self.conv1_WSI(x_WSI)   
        x_WSI1 = self.APool_1(x_WSI1)    
        
        b, c, s, k, k = x_WSI1.size()
        x_CT2 = x_CT1.view(b, c, t*h*w).permute(0,2,1)
        x_WSI2 = x_WSI1.view(b, c, s*k*k)
        multi = torch.bmm(x_CT2, x_WSI2)  
        feature = multi.view(b, t, h, w, s*k*k).permute(0,4,1,2,3)
        
        Att = self.sigmoid(self.conv1_Att(feature))    
        
        # 融合
        x_WSI3 = self.conv3_rr(x_WSI) 
        b, cr, s, h, w = x_WSI3.size()
        x_WSI4 = x_WSI3.view(b, cr*s, h, w)
        x_WSI5 = x_WSI4.unsqueeze(2).repeat(1,1,t,1,1)
        x_FS = torch.cat((x_CT, x_WSI5), 1)
        
        Att_CT = Att[:,0:1,:,:,:] 
        Att_CT = Att_CT.repeat(1,c,1,1,1) 
        
        
        Att_WSI = Att[:,1:,:,:,:].view(b,s,t,h*w).repeat(1,1,c//self.rr,1)  
        Att_WSI = Att_WSI.view(b, s*(c//self.rr), t, h, w)
        Att_FS  = torch.cat((Att_CT, Att_WSI), 1)
        
        x_Att = x_FS*Att_FS + x_FS

        Inv = self.conv1_Inv(feature) 
        Inv = Inv.permute(0,2,1,3,4).view(b, t, k, k, h, w) 
        x_Att1 = self.relu1(self.bn1(self.proj1(x_Att)))
        out = self.Inv(x_Att1, Inv).view(b, c+s*(c//self.rr), t, h, w)  
        out = self.proj2(self.bn2(out)+x_Att)                  
        return out 



