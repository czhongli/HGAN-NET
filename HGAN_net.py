import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor

class HSI_Generator(nn.Module):
    def __init__(self):
        super(HSI_Generator, self).__init__()
        
        # 光谱卷积分支
        self.spectral_conv_3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
        
        self.spectral_conv_5 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2)
        )
        
        self.spectral_conv_7 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3)
        )
        
        # 空间卷积分支
        self.spatial_conv_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.Conv2d(16, 8, kernel_size=3, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        self.spatial_conv_5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5, padding=2), 
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        self.spatial_conv_7 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=7, padding=3), 
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        # 噪声感知分支
        self.noise_sensing = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        
        # 门控信号分支
        self.gating = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 特征结合模块
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(120, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)  
        )
        
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
        
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
        
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1)
        )
        
        self.convcle = nn.Sequential(
            nn.Conv2d(129, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )
        
        
    def forward(self, input_spatial, input_spectral):
        input_spatial_3 = self.spectral_conv_3(input_spatial)
        input_spatial_5 = self.spectral_conv_5(input_spatial)
        input_spatial_7 = self.spectral_conv_7(input_spatial)
        x1 = torch.cat((input_spatial_3, input_spatial_5, input_spatial_7), dim=1)

        x_spatial_3 = self.spatial_conv_3(input_spectral)
        x_spatial_5 = self.spatial_conv_5(input_spectral)
        x_spatial_7 = self.spatial_conv_7(input_spectral)
        x2 = torch.cat((x_spatial_3, x_spatial_5, x_spatial_7), dim=1)

        x = torch.cat((x1, x2), dim=1) # f_IMF

        # 特征融合模块（逐层）
        x1 = self.conv_block_1(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        x4 = self.conv_block_4(x3)
        x = torch.cat((x1, x2, x3, x4, input_spatial), dim=1) # 残差结构
        x = self.convcle(x)

        x_noise = self.noise_sensing(input_spatial) # f_E
        gating_signal = self.gating(x_noise) #f_G
        
        x = torch.dot(x, gating_signal)
        clean = input_spatial+x # 残差学习  

        return clean



class HSI_Discriminator(nn.Module):
    def __init__(self):
        super(HSI_Discriminator, self).__init__()
        
        # 鉴别器部分
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), 
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(6 * 6, 1),# 6*6
            nn.Sigmoid()
        )
    
    def forward(self,cle):
        p = self.discriminator(cle) # 概率值
        
        return p