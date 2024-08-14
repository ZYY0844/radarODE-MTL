import torch, sys, os, math
sys.path.append(os.getcwd())
from Projects.radarODE_plus.utils.ODE.equations import d_x_d_t, d_y_d_t, d_z_d_t
from Projects.radarODE_plus.utils.ODE.ode_params import ODEParams
from torchinfo import summary
import torch.nn as nn


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # for model print only

default_input = torch.Tensor((
        ([5, 0.25, -15.0 * math.pi / 180.0,
          -100.0, 0.1, 25.0 * math.pi / 180.0,
          480.0, 0.1, 40.0 * math.pi / 180.0,
          -120, 0.1, 60.0 * math.pi / 180.0,
          8, 0.4, 135.0 * math.pi / 180.0]))).reshape((1, 15))

def ode1_solver(input_params, ecg_len=200):
    """
    ecg_len: length of ecg signal, determined by estimeated period
    input_params: parameters for PQRST, with size (batch_size, 15)
    return
    ecg_signal: with size batch_size * ecg_len
    """
    device = input_params.device
    ode_params = ODEParams(input_params.device, ecg_len)
    batchsize = input_params.shape[0]
    # initial values (do not change)
    x = torch.tensor(
        [-0.417750770388669 for _ in range(input_params.shape[0])]).view(batchsize, 1).to(device)
    y = torch.tensor(
        [-0.9085616622823985 for _ in range(input_params.shape[0])]).view(batchsize, 1).to(device)
    z = torch.tensor(
        [-0.004551233843726818 for _ in range(input_params.shape[0])]).view(batchsize, 1).to(device)
    t = torch.tensor([0.0 for _ in range(input_params.shape[0])]
                     ).reshape((-1, 1)).view(batchsize, 1).to(device)

    x_signal = [x]
    y_signal = [y]
    z_signal = [z]
    for i in range(ecg_len-1):
        x, y, z, t = x.to(device), y.to(device), z.to(device), t.to(device)
        f_x = d_x_d_t(y, x, t, ode_params.omega)
        f_y = d_y_d_t(y, x, t, ode_params.omega)
        f_z = d_z_d_t(x, y, z, t, input_params, ode_params)
        t += 1 / 200

        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z

    z_signal = torch.stack(z_signal)
    return z_signal.squeeze(2).permute(1, 0)

def scale_output(input_params):
    # use input_params as the percentage range to scale the default_input
    # default_input = torch.Tensor(default_input)
    input_params = (1+1*input_params) * default_input.to(input_params.device) # 2 is the factor for percentage

    return input_params


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

class ECGParameterEstimator(nn.Module):
    def __init__(self):
        super(ECGParameterEstimator, self).__init__()
        # Encoder layers
        # self.encoder = nn.Sequential(
        #     conv2DBlock(1, 16, kernel_size=5, stride=(2,2), padding=(1,1)),
        #     nn.Conv2d(16, 32, 3, stride=(1,2), padding=(1,1)),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(32, 64, 3, stride=(1,2), padding=(1,1)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=False),
        #     nn.Conv2d(64, 64, 1, stride=(2,1), padding=(0,0)),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=False)
        # )
        
        # Decoder layers
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 5, stride=(2,2), padding=(1,1), output_padding=1),
        #     nn.BatchNorm2d(32),
        #     # nn.Sigmoid(),
        #     nn.ReLU(inplace=False),
        #     nn.ConvTranspose2d(32, 16, 5, stride=(2,2), padding=1, output_padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=False),
        #     nn.ConvTranspose2d(16, 8, 3, stride=(1,2), padding=1, output_padding=0),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(inplace=False),
        #     nn.ConvTranspose2d(8, 4, 3, stride=(1,1), padding=1, output_padding=0),
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(inplace=False),
        #     nn.ConvTranspose2d(4, 1, 1, stride=(1,1), padding=1, output_padding=0),
        #     nn.ReLU(inplace=False),
        # )
        # self.decoder = nn.Sequential(
        #     ConvBlock(128, 64, kernel_size=7, stride=2, padding=1),
        #     ConvBlock(64, 32, kernel_size=7, stride=2, padding=1),
        #     ConvBlock(32, 16, kernel_size=7, stride=1, padding=1),
        #     ConvBlock(16, 8, kernel_size=5, stride=1, padding=1),

        # )

        # self.out_conv = nn.Sequential(
        #     nn.Conv1d(8, 4, kernel_size=7, stride=1, padding=1),
        #     nn.Conv1d(4, 2, kernel_size=5, stride=1, padding=1),
        #     nn.Conv1d(2, 1, kernel_size=4, stride=1, padding=0),
        #     nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),
        # )
        self.linear_out = nn.Sequential(
            nn.Linear(8*207, 1024,bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Linear(1024, 512,bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 256,bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128,bias=True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64,bias=True),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32,bias=True),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 16,bias=True),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, 15,bias=True),
            nn.Tanh(),
        )
        self.activation=nn.Tanh()
    def forward(self, x):
        # x = self.encoder(x)
        # x = self.decoder(x).squeeze(1)
        # x = self.out_conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_out(x)
        return x

if __name__ == '__main__':
    # input_dim=2001 # dimension of signal index from -1 to 1
    # hidden_dim = 80 # dimension of embedding
    # num_layers = 6
    # output_dim=15
    # nhead=8
    # embed_dim=hidden_dim #dim of the signal_vector
    batchsize=2
    input_shape = (batchsize, 8, 207)
    # Create the model
    # model = RegressionTransformer(input_dim, output_dim, nhead, hidden_dim, num_layers)
    model =  ECGParameterEstimator()
    inputs= torch.rand(input_shape)
    output = model(inputs)
    print(summary(model, input_size=input_shape, device='cpu'))
    print(output.min(), output.max())
 