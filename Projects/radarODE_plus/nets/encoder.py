import torch,sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch.nn as nn
import torch.nn.functional as F
import math
from deformable_attention import DeformableAttention1D
from einops import rearrange

from torchinfo import summary


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, dim):
        super().__init__()
        num_pos_feats = dim // 2
        self.row_embed = nn.Embedding(dim, num_pos_feats)
        self.col_embed = nn.Embedding(dim, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        x = tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


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


class TransformerEncoder(nn.Module):
    def __init__(self, dim, attn_type='mhsa'):
        super().__init__()
        self.dim = dim
        self.attn_type = attn_type

        self.pos_emb = PositionEmbeddingLearned(dim=dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, x):
        print(x.shape)
        x = x + self.pos_emb(x)
        b, c, h, w = x.shape
        x = torch.flatten(x, 2).permute(2, 0, 1)
        output = self.encoder(x)
        return output


class LSTMCNNEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # self.lstm1 = nn.LSTM(31, dim // 4, num_layers=1, batch_first=True, bidirectional=True)
        self.deconv = nn.Sequential(
            DeconvBlock(dim, dim // 2, kernel_size=5, stride=3, padding=0),
            DeconvBlock(dim // 2, dim // 4, kernel_size=5, stride=3, padding=0),
            DeconvBlock(dim // 4, dim // 8, kernel_size=5, stride=3, padding=0)
        )

    def forward(self, x):
        # x, _ = self.lstm1(x)
        # x = x.unsqueeze(1)
        x = self.deconv(x)

        return x



if __name__ == '__main__':
    model = LSTMCNNEncoder(1024)
    # Print the model summary to check the output dimensions
    print(model)

    # Test the model with a random input
    input_tensor = torch.randn(2, 1024, 31)  # Example input tensor
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected shape: (2, 128, 863)
    print(summary(model, input_size=(2, 1024, 31), device='cpu'))
