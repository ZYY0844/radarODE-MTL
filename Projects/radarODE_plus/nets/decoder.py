
import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
# from TCN_decoder import TemporalConvNet
BASE_DIR = (os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from nets.ODE_solver import ECGParameterEstimator, ode1_solver, scale_output


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


class CNNLSTMDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.ode_solver =  ode1_solver()
        self.param_estimator=ECGParameterEstimator()
        self.conv = nn.Sequential(
            ConvBlock(128, 64, kernel_size=7, stride=2, padding=1),
            ConvBlock(64, 32, kernel_size=7, stride=2, padding=1),
            ConvBlock(32, 16, kernel_size=7, stride=1, padding=1),
            ConvBlock(16, 8, kernel_size=5, stride=1, padding=1),

        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=7, stride=1, padding=1),
            nn.Conv1d(4, 2, kernel_size=5, stride=1, padding=1),
            nn.Conv1d(2, 1, kernel_size=2, stride=1, padding=0),
        )
        # fused feature Encoder layers
        self.fusion_encoder = nn.Sequential(
            conv2DBlock(1, 16, kernel_size=5, stride=(2,2), padding=(1,1)),
            conv2DBlock(16, 32, 3, stride=(1,2), padding=(1,1)),
            conv2DBlock(32, 64, 3, stride=(1,2), padding=(1,1)),
            conv2DBlock(64, 64, 1, stride=(2,1), padding=(0,0)),
        )
        
        # Decoder layers
        self.fusion_decoder = nn.Sequential(
            convTransBlock(64, 32, 5, stride=(2,2), padding=(1,1), output_padding=1),
            convTransBlock(32, 16, 5, stride=(2,2), padding=1, output_padding=1),
            convTransBlock(16, 8, 3, stride=(1,2), padding=1, output_padding=0),
            convTransBlock(8, 4, 3, stride=(1,1), padding=1, output_padding=0),
            convTransBlock(4, 1, 1, stride=(1,1), padding=1, output_padding=0),
        )

        self.out_conv_final = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=7, stride=1, padding=1),
            nn.Conv1d(4, 2, kernel_size=5, stride=1, padding=1),
            nn.Conv1d(2, 1, kernel_size=4, stride=1, padding=0),
            nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),
        )
        # for test only
        # self.fusion_encoder_test = nn.Sequential(
        #     ConvBlock(1, 8, kernel_size=5, stride=2, padding=1),
        #     ConvBlock(8, 16, 3, stride=2, padding=1),
        #     ConvBlock(16, 32, 3, stride=2, padding=1),
        #     ConvBlock(32, 64, 3, stride=1, padding=1),
        #     # ConvBlock(64, 64, 1, stride=, padding=0),
        # )

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
        # temp+ode, give the best result but time consuming due to the ode solver
        # x = self.conv(x)
        # x_temproal = self.out_conv(x)
        # x_temproal =  x_temproal.squeeze(1)
        # param = self.param_estimator(x)
        # x_ode = ode1_solver(scale_output(param))
        # x_fusion = x_temproal * x_ode
        # x_fusion = torch.stack([x_fusion, x_fusion, x_fusion, x_fusion], dim=1).unsqueeze(1)
        # x_fusion =  self.fusion_encoder(x_fusion)
        # x_final = self.fusion_decoder(x_fusion).squeeze(1)
        # x_final = self.out_conv_final(x_final)
        # return x_final
    
        # ### temp only, faster but not as good as temp+ode
        x = self.conv(x)
        x_temproal = self.out_conv(x)
        return x_temproal
    
        # ### ode only, for ablation study only
        # x = self.conv(x)
        # param = self.param_estimator(x)
        # x_ode = ode1_solver(scale_output(param))
        # return x_ode.unsqueeze(1)
    
        
class TransformerDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        decoder_layer = nn.TransformerDecoderLayer(d_model=1024, nhead=8)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=4)
        # ffn
        self.linear1 = nn.Linear(1, dim)
        self.activation1 = nn.Tanh()
        self.dropout1 = nn.Dropout(0.01)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(0.01)
        self.norm1 = nn.LayerNorm(dim)

        self.linear_last1 = nn.Linear(dim, dim//2)
        self.activation2 = nn.Tanh()
        self.linear_last2 = nn.Linear(dim//2, 1)

    def forward(self, src, tgt=None):
        tgt_new = self.norm1(self.dropout2(self.linear2(
            self.dropout1(self.activation1(self.linear1(tgt))))))
        output = self.decoder(tgt_new, src)
        output = self.linear_last2(self.activation2(
            self.linear_last1(output))).permute(1, 2, 0)
        return output
    
if __name__ == '__main__':
    model = CNNLSTMDecoder()
    # input_dim = 2001  # dimension of signal index from -1 to 1
    # hidden_dim = 80  # dimension of embedding
    # num_layers = 6
    # output_dim = 1
    # embed_dim = hidden_dim
    # nhead = 8
    # model = RegressionTransformer(
    #     input_dim, output_dim, nhead, hidden_dim, embed_dim, num_layers)
    input_shape = (2, 128, 863)
    input_data = torch.randn(input_shape)
    output_data = model(input_data)
    print(output_data.shape)
    print(summary(model, input_size=input_shape, device='cpu'))
