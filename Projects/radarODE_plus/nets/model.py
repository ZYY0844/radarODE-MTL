import torch, sys, os
import torch.nn as nn

sys.path.append(os.getcwd())
from Projects.radarODE_plus.nets.backbone.dcnresnet_backbone import DCNResNet
from Projects.radarODE_plus.nets.backbone.squeeze_module import SqueezeModule
from Projects.radarODE_plus.nets.encoder import LSTMCNNEncoder
from Projects.radarODE_plus.nets.decoder import CNNLSTMDecoder
from Projects.radarODE_plus.nets.PPI_decoder import PPI_decoder

from utils import global_var

from torchinfo import summary

class backbone(nn.Module):
    def __init__(self, in_channels=50):
        super().__init__()
        self.backbone = DCNResNet(in_channels)
        self.squeeze_module = SqueezeModule(in_channels=1024, out_channels=1024)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        img_feature = self.backbone(x)
        img_feature = self.squeeze_module(img_feature)
        return img_feature
    
class shapeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTMCNNEncoder(dim=1024)
        self.decoder = CNNLSTMDecoder()
    def forward(self, series_feature):
        ecg_shape = self.encoder(series_feature)
        ecg_shape = self.decoder(ecg_shape)
        return ecg_shape

# !!!!!
# Only for illustration purposes, not used anywhere in the code 
# !!!!!
class ECGFormer(nn.Module):
    def __init__(self, in_channels=50, model_type='v1'):
        super().__init__()

        self.in_channels = in_channels

        self.backbone = DCNResNet(in_channels)
        self.squeeze_module = SqueezeModule(in_channels=1024, out_channels=1024)
        self.encoder = LSTMCNNEncoder(dim=1024)
        self.decoder = CNNLSTMDecoder()
        self.PPI_decoder = PPI_decoder(output_dim=260)
        self.anchor_decoder = PPI_decoder(output_dim=800)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, tgt=None):
        img_feature = self.backbone(x)
        img_feature = self.squeeze_module(img_feature)
        # ppi estimation
        ppi = self.PPI_decoder(img_feature)
        # anchor estimation
        anchor = self.anchor_decoder(img_feature)
        # ecg reconstruction
        series_feature = self.encoder(img_feature)
        ecg_shape = self.decoder(series_feature)
        
        return ppi, anchor, ecg_shape

if __name__ == '__main__':
    model = ECGFormer().cuda()
    input_data = torch.randn(2, 50, 71, 120).cuda()
    tgt_data = torch.randn(200, 1, 1).cuda()
    model.eval()
    output = model(input_data)
    print("output shape:", output[0].shape, output[1].shape, output[2].shape)
    print(summary(model, input_size=[(2, 50, 71, 120)]))
    # a=1

