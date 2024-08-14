import torch
import torch.nn as nn


class SqueezeModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(6, 1), stride=1)

    def forward(self, x):
        output = self.squeeze(x)
        output = torch.squeeze(output, 2)
        return output


if __name__ == '__main__':
    model = SqueezeModule(in_channels=1024, out_channels=2048).cuda()
    input_data = torch.randn(1, 1024, 6, 31).cuda()
    output = model(input_data)
    print(output.shape)


