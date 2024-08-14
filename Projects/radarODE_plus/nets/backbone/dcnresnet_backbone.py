import torch,os,sys
import torch.nn as nn
sys.path.append(os.getcwd())
from Projects.radarODE_plus.nets.backbone.dcnv2 import DeformConv2d


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.dcn_conv1 = DeformConv2d(inc=in_channels, outc=in_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=False)
        self.dcn_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding)

    def forward(self, x):
        output = self.dcn_conv1(x)
        output = self.bn(output)
        output = self.activation(output)
        output = output + x

        output = self.dcn_conv2(output)

        return output


class Downsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.dcn_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, x):
        output = self.activation(self.bn(self.dcn_conv(x)))
        return output


class DCNResNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stage_channels = [in_channels, 128, 256, 512, 1024]

        self.stage0 = BasicBlock(self.stage_channels[0], self.stage_channels[1], kernel_size=(2, 1), stride=(1, 1),
                                 padding=(1, 0))
        self.downsample0 = Downsampling(in_channels=self.stage_channels[1], out_channels=self.stage_channels[1],
                                        kernel_size=(3, 2), stride=2, padding=1)

        self.stage1 = BasicBlock(self.stage_channels[1], self.stage_channels[2], kernel_size=(2, 1), stride=(1, 1),
                                 padding=(1, 0))
        self.downsample1 = Downsampling(in_channels=self.stage_channels[2], out_channels=self.stage_channels[2],
                                        kernel_size=(3, 2), stride=2, padding=1)

        self.stage2 = BasicBlock(self.stage_channels[2], self.stage_channels[3], kernel_size=(2, 1), stride=(1, 1),
                                 padding=(1, 0))
        self.downsample2 = Downsampling(in_channels=self.stage_channels[3], out_channels=self.stage_channels[3],
                                        kernel_size=(3, 3), stride=(2, 1), padding=1)

        self.stage3 = BasicBlock(self.stage_channels[3], self.stage_channels[4], kernel_size=(2, 1), stride=(1, 1),
                                 padding=(1, 0))
        self.downsample3 = Downsampling(in_channels=self.stage_channels[4], out_channels=self.stage_channels[4],
                                        kernel_size=(3, 3), stride=(2, 1), padding=1)


    def forward(self, x):
        x = self.stage0(x)
        x = self.downsample0(x)

        x = self.stage1(x)
        x = self.downsample1(x)

        x = self.stage2(x)
        x = self.downsample2(x)

        x = self.stage3(x)
        x = self.downsample3(x)

        return x


if __name__ == '__main__':
    model = DCNResNet(in_channels=50)
    input_map = torch.randn(2, 50, 71, 120)
    output_map = model(input_map)
    print(output_map.shape)