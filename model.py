import torch
import numpy as np

from torch import nn

# U-Net

# Two convolution block. Performs two consecutive convolutions
class TwoConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__()

        self.module_list = nn.ModuleList([])
        
        #Using Henriks convultion layering or the one introduced in the Unet paper?
        self.module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.module_list.append(nn.ReLU())

        self.module_list.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.module_list.append(nn.ReLU())

    def forward(self, x):
        y = x
        for module in self.module_list:
            y = module(y)
        return y

# UNet encoder block. Performs two convolutions and max pooling.
class ConvPool(TwoConv):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.max = nn.MaxPool2d(2, 2)

    def forward(self, x):
        c = super().forward(x)
        p = self.max(c)
        return c, p

# UNet decoder block. Performs upsampling, concatenation of the two inputs and two convolutions.
class UpConv(TwoConv):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        # We may use different upsampling method here.
        self.upsampling = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        u = self.upsampling(x)
        u = torch.cat([u, skip], 0)
        c = super().forward(u)
        return c, u


class UNet(nn.Module):
    def __init__(self, in_channels, min, max, num_classes):
        super().__init__()
        self.enc_layers = nn.ModuleList([])
        self.dec_layers = nn.ModuleList([])
        self.enc_final = None
        self.dec_final = None
        self.softmax = None

        # When go down the encoder/up the decoder the number of filter doubles/halves
        # respectively. For that we will generate the powers of two.
        # List of powers of 2 [min, 2*min, 4*min, ..., max]
        channels = []
        power = min
        for i in range(int(np.log2(max // min))):
            channels.append(power)
            power = power*2

        # Construct list of blocks for the encoder
        self.enc_layers.append(ConvPool(in_channels, min))
        for i in range(len(channels)-1):
            enc_layer = ConvPool(channels[i], channels[i+1])
            self.enc_layers.append(enc_layer)

        # Construct list of blocks for the encoder
        for i in range(len(channels)-1):
            dec_layer = UpConv(channels[i+1], channels[i])
            self.dec_layers.insert(0, dec_layer)
        self.dec_layers.insert(0, UpConv(max, channels[-1]))

        # Set up final convolutions for the encoder and decoder
        self.enc_final = TwoConv(channels[len(channels)-1], max, 3, 1, 'same')
        self.dec_final = nn.Conv2d(min, num_classes, 1, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        # Collect the values for skip connections to the decoder
        skip_connections = []
        p = x
        # Encoder
        for layer in self.enc_layers:
            c, p = layer(p)
            skip_connections.append(c)

        # Bottleneck
        c =  self.enc_final(p)

        # Decoder
        for layer in self.dec_layers:
            skip = skip_connections.pop()
            c, u = layer(c, skip) # if we do not need c we can use _ instead
        c = self.dec_final(c)

        return self.softmax(c)