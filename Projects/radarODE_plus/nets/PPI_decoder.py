
import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
sys.path.append(os.getcwd())

from Projects.radarODE_plus.nets.ODE_solver import ECGParameterEstimator, ode1_solver, scale_output


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

        self.conv_encoder = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
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

# not only for ppi but also for anchor decoder with differnet output_dim
class PPI_decoder(nn.Module):
    def __init__(self, output_dim, dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(dim, dim//4, kernel_size=7, stride=1, padding=1),
            ConvBlock(dim//4, dim//8, kernel_size=7, stride=1, padding=1),
            ConvBlock(dim//8, dim//32, kernel_size=7, stride=1, padding=1),
            ConvBlock(dim//32, dim//64, kernel_size=6, stride=1, padding=1),
        )
        self.transconv = nn.Sequential(
            convTransBlock(dim//64, dim//64, kernel_size=5, stride=2, padding=2, output_padding=1),
            convTransBlock(dim//64, dim//32, kernel_size=5, stride=2, padding=2, output_padding=1),
            convTransBlock(dim//32, dim//32, kernel_size=5, stride=1, padding=2, output_padding=0),
            # convTransBlock(dim//4, dim, kernel_size=7, stride=1, padding=1, output_padding=0),
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Linear(512, output_dim),
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
        # all
        x = x.squeeze(2)
        x = self.conv(x)
        x = self.transconv(x)
        x = x.view(x.size(0), -1)
        x = self.prediction(x)
        return x.unsqueeze(1)
    
        
        
    
if __name__ == '__main__':
    model = PPI_decoder(output_dim=800)
    # input_dim = 2001  # dimension of signal index from -1 to 1
    # hidden_dim = 80  # dimension of embedding
    # num_layers = 6
    # output_dim = 1
    # embed_dim = hidden_dim
    # nhead = 8
    # model = RegressionTransformer(
    #     input_dim, output_dim, nhead, hidden_dim, embed_dim, num_layers)
    input_shape = (2, 1024, 1, 31)
    input_data = torch.randn(input_shape)
    output_data = model(input_data)
    print(output_data.shape)
    print(summary(model, input_size=input_shape, device='cpu'))
