
import torch
import os
import sys
sys.path.append(os.getcwd())
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
# from Projects.radarODE_plus.nets.encoder import LSTMCNNEncoder



class conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(conv2DBlock, self).__init__()

        self.conv_encoder = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
class convTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(convTransBlock, self).__init__()

        self.conv_encoder = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DeconvBlock, self).__init__()

        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x

class anchor_decoder(nn.Module):
    def __init__(self, dim = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            DeconvBlock(dim, dim // 2, kernel_size=5, stride=3, padding=2),
            DeconvBlock(dim // 2, dim // 4, kernel_size=5, stride=3, padding=2),
            DeconvBlock(dim // 4, dim // 8, kernel_size=5, stride=3, padding=2),
            DeconvBlock(dim // 8, dim // 8, kernel_size=5, stride=2, padding=2)
        )
        self.conv = nn.Sequential(
            ConvBlock(128, 64, kernel_size=5, stride=2, padding=2),
            ConvBlock(64, 32, kernel_size=5, stride=1, padding=2),
            ConvBlock(32, 16, kernel_size=5, stride=1, padding=2),
            # ConvBlock(16, 8, kernel_size=5, stride=1, padding=1),
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=5, stride=1, padding=1),
            nn.Conv1d(8, 4, kernel_size=5, stride=1, padding=0),
            nn.Conv1d(4, 2, kernel_size=5, stride=1, padding=0),
            nn.Conv1d(2, 1, kernel_size=2, stride=1, padding=0),
        )
    def _initialize_weights(self):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                # 判断是否有偏置
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zeros_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.conv(x)
        x = self.out_conv(x)
        return x


    
    
if __name__ == '__main__':
    model = anchor_decoder()
    input_shape = (2, 1024, 31)
    # input_shape = (2, 128, 863)
    input_data = torch.randn(input_shape)
    output_data = model(input_data)
    print(output_data.shape)
    # print(summary(model, input_size=input_shape, device='cpu'))
