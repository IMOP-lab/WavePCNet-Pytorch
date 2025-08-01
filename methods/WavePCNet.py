from torch import nn
from torch import Tensor
import torch
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np

custom_config = {'base'      : {'strategy': 'adam_base',
                                'batch': 8,
                               },
                 'customized': {'--abc': {'type': float, 'default': 0},
                                '--abc_true': {'action': 'store_true'},
                               },
                }

    
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        resnet = resnet50(pretrained=True)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)        
        self.layer2 = resnet.layer2                                      
        self.layer3 = resnet.layer3                                   
        self.layer4 = resnet.layer4 

    def forward(self, x):
        x1 = self.layer0(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5
    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class FGA(nn.Module):
    def __init__(self, channel):
        super(FGA, self).__init__()
        self.atrconv1 = BasicConv2d(channel, channel, 3, padding=3, dilation=3)
        self.atrconv2 = BasicConv2d(channel, channel, 3, padding=5, dilation=5)
        self.atrconv3 = BasicConv2d(channel, channel, 3, padding=7, dilation=7)

        self.fuse = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.branch1 = BasicConv2d(channel, channel, 3, padding=1, dilation=1)
        self.branch2 = BasicConv2d(channel, channel, 3, padding=3, dilation=3)
        self.branch3 = BasicConv2d(channel, channel, 3, padding=5, dilation=5)

        self.conv_cat1 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_cat2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_cat3 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv1_1 = BasicConv2d(channel, channel, 1)

        self.ca1 = ChannelAttention(channel)
        self.ca2 = ChannelAttention(channel)
        self.sa = SpatialAttention()
        self.edg_pred = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.sal_conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        fused = self.fuse(torch.cat([x1, b2, b3], dim=1))

        x_atr1 = self.atrconv1(x)
        s_mfeb1 = self.conv_cat1(torch.cat((x1, x_atr1), 1)) + x
        x2 = self.branch2(s_mfeb1)
        x_atr2 = self.atrconv2(s_mfeb1)
        s_mfeb2 = self.conv_cat2(torch.cat((x2, x_atr2), 1)) + s_mfeb1 + x
        x3 = self.branch3(s_mfeb2)
        x_atr3 = self.atrconv3(s_mfeb2)
        s_mfeb3 = self.conv_cat3(torch.cat((x3, x_atr3), 1)) + s_mfeb1 + s_mfeb2 + x
        s_mfeb = self.conv1_1(s_mfeb3)
        s_ca = self.ca1(s_mfeb) * s_mfeb

        e_input = fused + s_ca
        e_pred = self.edg_pred(e_input)
        s_mea = self.sal_conv((self.sa(s_ca) + self.sigmoid(e_pred)) * s_ca) + s_mfeb1 + s_mfeb2 + s_mfeb3 + x

        return s_mea

# Dynamic Cross-Attention Fusion
class DCAF(nn.Module):
    def __init__(self, channel):
        super(DCAF, self).__init__()
        
        self.query_conv = nn.Conv2d(channel, channel, 1)
        self.key_conv = nn.Conv2d(channel, channel, 1)
        self.value_conv = nn.Conv2d(channel, channel, 1)

        self.fh_conv = BasicConv2d(channel, channel, 3, padding=1)
        self.fl_conv = BasicConv2d(channel, channel, 3, padding=1)
        self.S_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

    def forward(self, fl, fh, f5, *extra_feats):

        fgl = F.interpolate(f5, size=fl.size()[2:], mode='bilinear')
        for f in extra_feats:
            f = F.interpolate(f, size=fl.size()[2:], mode='bilinear')
            fgl = fgl * f 

        query = self.query_conv(fgl)
        key_fh = self.key_conv(fh)
        key_fl = self.key_conv(fl)

        attn_fh = torch.sigmoid(query * key_fh)
        attn_fl = torch.sigmoid(query * key_fl)

        fh = self.fh_conv(fh * attn_fh) + fh
        fl = self.fl_conv(fl * attn_fl) + fl

        out = self.S_conv(torch.cat((fh, fl), 1))
        return out




class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StrongResNet(nn.Module):
    def __init__(self, channels=3, mid_channels=64, num_blocks=8):
        super().__init__()
        layers = [nn.Conv2d(channels, mid_channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(mid_channels, mid_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(mid_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(mid_channels, channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PhysicalDiffraction(nn.Module):
    def __init__(self, H_real, H_imag):
        super().__init__()
        # H_real, H_imag shape: (1, 3, H, W)
        self.H_real = nn.Parameter(H_real, requires_grad=False)
        self.H_imag = nn.Parameter(H_imag, requires_grad=False)

    def forward(self, x):
        x_fft = torch.fft.fft2(x)
        H_complex = torch.complex(self.H_real, self.H_imag)
        y_fft = x_fft * H_complex
        y = torch.fft.ifft2(y_fft)
        return y.real

class PhiNet(nn.Module):
    def __init__(self, H_real, H_imag):
        super().__init__()
        self.physical_layer = PhysicalDiffraction(H_real, H_imag)
        self.residual_net = StrongResNet(channels=3, mid_channels=64, num_blocks=8)

    def forward(self, x):
        Ax = self.physical_layer(x)
        res = self.residual_net(x)
        return Ax + res
    
# OTF 初始化函数
def generate_otf(H, W, wavelength=532e-9, z=0.5, pixel_size=1e-6):
    """ 生成简化菲涅尔衍射 OTF """
    fx = np.fft.fftfreq(W, d=pixel_size)
    fy = np.fft.fftfreq(H, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    k = 2 * np.pi / wavelength
    H_complex = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    H_real = np.real(H_complex).astype(np.float32)
    H_imag = np.imag(H_complex).astype(np.float32)
    H_real = torch.from_numpy(H_real).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # (1,3,H,W)
    H_imag = torch.from_numpy(H_imag).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    return H_real, H_imag


class Network(nn.Module):
    def __init__(self, config, encoder, feat):

        super(Network, self).__init__()
        # ------------------------ # 
        H_real, H_imag = generate_otf(H=256, W=256)
        self.phi_1 = PhiNet(H_real, H_imag)
        self.phit_1 = PhiNet(H_real, -H_imag)
        self.beta1 = 0.9
        self.beta2 = 0.999

        self.epsilon = 1e-8
        self.lr = nn.Parameter(torch.tensor(1e-4))
        self.r1 = nn.Parameter(torch.Tensor([0.5]))  
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # ------------------------ #         
        channel = 32
        self.encoder = ResNetBackbone()
        self.reduce_sal1 = Reduction(64, channel)     # x1
        self.reduce_sal2 = Reduction(256, channel)    # x2
        self.reduce_sal3 = Reduction(512, channel)    # x3
        self.reduce_sal4 = Reduction(1024, channel)   # x4
        self.reduce_sal5 = Reduction(2048, channel)   # x5

        self.fga5 = FGA(channel)
        self.fga4 = FGA(channel)
        self.fga3 = FGA(channel)
        self.fga2 = FGA(channel)
        self.fga1 = FGA(channel)

        self.dcaf1 = DCAF(channel)
        self.dcaf2 = DCAF(channel)
        self.dcaf3 = DCAF(channel)

        self.S1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.S_conv1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.trans_conv1 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        self.trans_conv2 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv3 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv4 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, phase='test'):
        # --------------------------- #
        # 1
        # --------------------------- #
        grad_1 = self.phi_1(x) - x
        x1_img = x - self.r1*self.phit_1(grad_1)       
        # # # --------------------------- #
        # 2
        # --------------------------- #
        m1 = grad_1.clone()
        v1 = grad_1.pow(2).clone()
        x2 = x1_img
        beta1 = self.beta1
        beta2 = self.beta2
        grad_2 = self.phi_1(x2) - x
        m2 = beta1 * m1 + (1 - beta1) * self.phit_1(grad_2)
        v2 = beta2 * v1 + (1 - beta2) * self.phit_1(grad_2).pow(2)

        m2_hat = m2 / (1 - beta1 ** 2)
        v2_hat = v2 / (1 - beta2 ** 2)
        m2_hat = torch.clamp(m2_hat, -1.0, 1.0)
        v2_hat = torch.clamp(v2_hat, min=1e-6)
        update = self.lr * m2_hat / (v2_hat.sqrt() + self.epsilon)
        update = torch.clamp(update, -10.0, 10.0)
        x2_img = x2 - update
        # -------------- #
        x3 = x2_img
        grad_3 = self.phi_1(x3) - x
        m3 = beta1 * m2 + (1 - beta1) * self.phit_1(grad_3)
        v3 = beta2 * v2 + (1 - beta2) * self.phit_1(grad_3).pow(2)

        m3_hat = m3 / (1 - beta1 ** 3)
        v3_hat = v3 / (1 - beta2 ** 3)
        m3_hat = torch.clamp(m3_hat, -1.0, 1.0)
        v3_hat = torch.clamp(v3_hat, min=1e-6)
        update = self.lr * m3_hat / (v3_hat.sqrt() + self.epsilon)
        update = torch.clamp(update, -10.0, 10.0)
        x3_img = x3 - update
        # -------------- #
        x_encoder_input = self.alpha * x + (1 - self.alpha) * x3_img

        size = x3_img.size()[2:]
        x_sal1, x_sal2, x_sal3, x_sal4, x_sal5 = self.encoder(x_encoder_input)

        x_sal1 = self.reduce_sal1(x_sal1)
        x_sal2 = self.reduce_sal2(x_sal2)
        x_sal3 = self.reduce_sal3(x_sal3)
        x_sal4 = self.reduce_sal4(x_sal4)
        x_sal5 = self.reduce_sal5(x_sal5)

        sal5 = self.fga5(x_sal5)
        sal4 = self.fga4(x_sal4)
        sal3 = self.fga3(x_sal3)
        sal2 = self.fga2(x_sal2)
        sal1 = self.fga1(x_sal1)

        sal4 = self.S_conv1(torch.cat([sal4, self.trans_conv1(sal5)], dim=1))
        sal3 = self.dcaf1(sal3, self.trans_conv2(sal4), sal5)
        sal2 = self.dcaf2(sal2, self.trans_conv3(sal3), sal5, sal4)
        sal1 = self.dcaf3(sal1, self.trans_conv4(sal2), sal5, sal4, sal3)

        sal_out = self.S1(sal1)
        sal2 = self.S2(sal2)
        sal3 = self.S3(sal3)
        sal4 = self.S4(sal4)
        sal5 = self.S5(sal5)

        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        sal2 = F.interpolate(sal2, size=size, mode='bilinear', align_corners=True)
        sal3 = F.interpolate(sal3, size=size, mode='bilinear', align_corners=True)
        sal4 = F.interpolate(sal4, size=size, mode='bilinear', align_corners=True)
        sal5 = F.interpolate(sal5, size=size, mode='bilinear', align_corners=True)

        OutDict = {
            'final': sal_out,
            'sal': [sal_out, sal2, sal3, sal4, sal5],
        }
        return OutDict