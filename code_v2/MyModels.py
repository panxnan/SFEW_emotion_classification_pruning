"""
constructing ResNet module

reference: 
https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

from torch.nn.modules.activation import LeakyReLU, ReLU, Softmax
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d, ConvTranspose2d
from torchvision import models
import torch
import torch.nn as nn
import torch.functional as F

# load pretrained model from pytorch
def Resnet18(pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    model = modify_model(model)
    return model

def Resnet50(pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    model = modify_model(model)
    return model

def Resnext50_32x4d(pretrained=True):
    model = models.resnext50_32x4d(pretrained=pretrained)
    model = modify_model(model)
    return model

# modify the last layer of pretrained model -> transfer learning
def modify_model(model):
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 7),
        nn.Softmax(dim=1)
    )
    return model



class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        # encoder layers
        # input is 256 * 256 *3
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros')
        self.conv1 = self._make_encoder(3, 8)
        self.pool1 = nn.MaxPool2d(9, stride = 1, return_indices=True)
        self.conv2 = self._make_encoder(8, 16)
        self.pool2 = nn.MaxPool2d(9, stride = 1, return_indices=True)

        self.encFC1 = nn.Linear(16*46*46, 4096)
        self.encFC2 = nn.Linear(16*46*46, 4096)
        self.decFC1 = nn.Linear(4096, 16*46*46)

        # self.trans_conv1 = self._make_decoder(32, 16)
        self.trans_conv1 = self._make_decoder(16, 8)
        self.unpool1 = nn.MaxUnpool2d(9, stride=1)
        self.unpool2 = nn.MaxUnpool2d(9, stride=1)
        self.trans_conv2 = self._make_decoder_final(8, 3)

        
    def _make_encoder(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=9, stride=2),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True),     
        )

    def _make_decoder(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=9, stride=2, output_padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.1, inplace=True)
        )
    
    def _make_decoder_final(self, in_channel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=9, stride=2, output_padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x):
        x = self.conv1(x)
        size1 = x.size()
        x, idx1 = self.pool1(x)

        x = self.conv2(x)
        size2 = x.size()
        x, idx2 = self.pool2(x)
        
        x = x.view(-1, 16*46*46)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        z = self.reparameterize(mu, logVar)

        x = self.decFC1(z)
        x = x.view(-1, 16, 46, 46)

        x = self.unpool1(x, idx2, output_size = size2)
        x = self.trans_conv1(x)
        x = self.unpool1(x, idx1, output_size = size1)
        x = self.trans_conv2(x)
        return x, mu, logVar