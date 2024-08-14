import torch,sys,os
import logging
import math
sys.path.append(os.getcwd())
from Projects.radarODE_plus.utils.ODE.ode_params import ODEParams
import time
from matplotlib import pyplot as plt
import numpy as np


def d_x_d_t(y, x, t, omega):
    alpha = 1 - (x**2 + y**2) ** 0.5
    f_x = alpha * x - omega * y
    return f_x

def d_y_d_t(y, x, t, omega):
    alpha = 1 - (x**2 + y**2) ** 0.5
    f_y = alpha * y + omega * x
    return f_y

def d_z_d_t(x, y, z, t, params, ode_params):
    """

    :param x:
    :param y:
    :param z:
    :param t:
    :param params:
    :param ode_params: Nx15
    :return:
    """
    A = ode_params.A
    f2 = ode_params.f2
    params=params.to(x.device)
    a_p, a_q, a_r, a_s, a_t = params[:, 0].view(-1, 1), params[:, 3].view(-1, 1), params[:, 6].view(-1, 1), params[:, 9].view(-1, 1), params[:, 12].view(-1, 1)

    b_p, b_q, b_r, b_s, b_t = params[:, 1].view(-1, 1), params[:, 4].view(-1, 1), params[:, 7].view(-1, 1), params[:, 10].view(-1, 1), params[:, 13].view(-1, 1)

    theta_p, theta_q, theta_r, theta_s, theta_t = params[:, 2].view(-1, 1), params[:, 5].view(-1, 1), params[:, 8].view(-1, 1), params[:, 11].view(-1, 1), params[:, 14].view(-1, 1)

    logging.debug("theta p shape: {}".format(theta_p.shape))
    theta = torch.atan2(y, x).to(x.device)
    logging.debug("theta shape: {}".format(theta.shape))
    # logging.debug("delta before mod: {}".format((theta - theta_p).shape))
    delta_theta_p = torch.fmod(theta - theta_p, 2 * math.pi)
    logging.debug("delta theta shape: {}".format(delta_theta_p.shape))
    delta_theta_q = torch.fmod(theta - theta_q, 2 * math.pi)
    delta_theta_r = torch.fmod(theta - theta_r, 2 * math.pi)
    delta_theta_s = torch.fmod(theta - theta_s, 2 * math.pi)
    delta_theta_t = torch.fmod(theta - theta_t, 2 * math.pi)

    z_p = a_p * delta_theta_p * \
          torch.exp((- delta_theta_p * delta_theta_p / (2 * b_p * b_p)))

    z_q = a_q * delta_theta_q * \
          torch.exp((- delta_theta_q * delta_theta_q / (2 * b_q * b_q)))

    z_r = a_r * delta_theta_r * \
          torch.exp((- delta_theta_r * delta_theta_r / (2 * b_r * b_r)))

    z_s = a_s * delta_theta_s * \
          torch.exp((- delta_theta_s * delta_theta_s / (2 * b_s * b_s)))

    z_t = a_t * delta_theta_t * \
          torch.exp((- delta_theta_t * delta_theta_t / (2 * b_t * b_t)))

    z_0_t = (A * torch.sin(2 * math.pi * f2 * t))

    z_p = z_p.to(x.device)
    z_q = z_q.to(x.device)
    z_r = z_r.to(x.device)
    z_s = z_s.to(x.device)
    z_t = z_t.to(x.device)
    z_0_t = z_0_t.to(x.device)

    f_z = -1 * (z_p + z_q + z_r + z_s + z_t) - (z - z_0_t)
    return f_z

# for test only
def test_equations_on_batch():
    device = 'cpu'
    ode_params = ODEParams(device)

    input = np.array((
             ([1.2, 0.25, -60.0 * math.pi / 180.0, -5.0, 0.1, -15.0 * math.pi / 180.0,
                           30.0, 0.1, 0.0 * math.pi / 180.0, -7.5, 0.1, 15.0 * math.pi / 180.0, 0.75, 0.4,
                           90.0 * math.pi / 180.0]))).reshape((1,15))
    a1 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    a2 = torch.nn.Parameter(torch.Tensor(input), requires_grad=True)
    print(a1.shape)
    input_params = torch.cat((a1, a2), 0)
    print("Input params shape: {}".format(input_params.shape))
    x = torch.tensor([-0.417750770388669, -0.417750770388669]).view(2, 1).to(device)
    y = torch.tensor([-0.9085616622823985, -0.9085616622823985]).view(2, 1).to(device)
    z = torch.tensor([-0.004551233843726818, 0.03]).view(2, 1).to(device)
    t = torch.tensor(0.0).to(device)

    x_signal = [x]
    y_signal = [y]
    z_signal = [z]
    start = time.time()
    for i in range(215):
        f_x = d_x_d_t(y, x, t, ode_params.omega)
        f_y = d_y_d_t(y, x, t, ode_params.omega)
        f_z = d_z_d_t(x, y, z, t, input_params, ode_params)
        t += 1 / 512
        # logging.info("f_z shape: {}".format(f_z.shape))
        # logging.info("f_y shape: {}".format(f_y.shape))
        # logging.info("f_x shape: {}".format(f_x.shape))
        x_signal.append(x + ode_params.h * f_x)
        y_signal.append(y + ode_params.h * f_y)
        z_signal.append(z + ode_params.h * f_z)

        x = x + ode_params.h * f_x
        y = y + ode_params.h * f_y
        z = z + ode_params.h * f_z
    end = time.time()

    logging.info("time: {}".format(end - start))
    z_signal = torch.stack(z_signal)
    logging.info('z_signal shape: {}'.format(z_signal.shape))
    res = [v[0].detach().numpy() for v in z_signal]

    print(len(res))
    print(res[0])
    plt.plot(res)
    plt.show()

    res = [v[1].detach().numpy() for v in z_signal]

    print(len(res))
    print(res[0])
    plt.plot(res)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_equations()
    test_equations_on_batch()