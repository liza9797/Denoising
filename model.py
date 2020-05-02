import torch 
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size=3, dilation=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
        nn.PReLU()
    )

def deconv_block(in_channels, out_channels, kernel_size=3, dilation=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, dilation=dilation),
        nn.PReLU()
    )

class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_down1 = conv_block(in_channels=1, out_channels=16, 
                                     kernel_size=(3, 5), dilation=1, padding=(1, 2))
        self.conv_down2 = conv_block(in_channels=16, out_channels=32, 
                                     kernel_size=(3, 5), dilation=3)
        self.conv_down3 = conv_block(in_channels=32, out_channels=64, 
                                     kernel_size=(3, 5), dilation=3)
        self.conv_down4 = conv_block(in_channels=64, out_channels=128, 
                                     kernel_size=(3, 5), dilation=6)
        
        self.deconv_up1 = deconv_block(in_channels=128, out_channels=64, kernel_size=(3, 5), dilation=6)
        self.conv_up1 = conv_block(in_channels=64, out_channels=64, kernel_size=3, dilation=1, padding=1)
        self.deconv_up2 = deconv_block(in_channels=64, out_channels=32, kernel_size=(3, 5), dilation=3)
        self.conv_up2 = conv_block(in_channels=32, out_channels=32, kernel_size=3, dilation=1, padding=1)
        self.deconv_up3 = deconv_block(in_channels=32, out_channels=16, kernel_size=(3, 5), dilation=3)
        self.conv_up3 = conv_block(in_channels=16, out_channels=16, kernel_size=3, dilation=1, padding=1)
        
        self.conv_last = conv_block(in_channels=16, out_channels=1, kernel_size=1, dilation=1)
        
    def forward(self, x):

        x1 = self.conv_down1(x)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)
        x = self.conv_down4(x3)
        
        x = self.deconv_up1(x)
        x = self.conv_up1(x + x3)
        
        x = self.deconv_up2(x)
        x = self.conv_up2(x + x2)
        
        x = self.deconv_up3(x)
        x = self.conv_up3(x + x1)
        
        x = self.conv_last(x)

        
        return x
    
    
    