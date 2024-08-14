import torch, sys, math
# from ODE import utils


class ODEParams:
    def __init__(self, device_name, ecg_length=216):
        self.A = torch.tensor(0.005).to(device_name)  # mV
        self.f1 = torch.tensor(0.1).to(device_name)  # mean 1
        self.f2 = torch.tensor(0.25).to(device_name)  # mean 2
        self.c1 = torch.tensor(0.01).to(device_name)  # std 1
        self.c2 = torch.tensor(0.01).to(device_name)  # std 2
        self.omega= torch.tensor(2.0 * math.pi / 1.0011).to(device_name) # the angular frequency of each cardiac cycle (one piece of ECG)
        # self.rrpc = utils.generate_omega_function(self.f1, self.f2, self.c1, self.c2)
        # self.rrpc = torch.tensor(self.rrpc).to(device_name).float()
        #T=216 is the sampl points for one heartbeat (600ms with sampling freq=1/360), h is the step-size for runge-kutta or Euler method
        self.h = torch.tensor(1 / ecg_length).to(device_name) 
    def para2numpy(self):
        self.A=self.A.numpy()
        self.f1=self.f1.numpy()
        self.f2=self.f2.numpy()
        self.c1=self.c1.numpy()
        self.c2=self.c2.numpy()
        self.omega=self.omega.numpy()
        self.rrpc=self.rrpc.numpy()
        self.h=self.h.numpy()
        return self
    
def index_convert(index, freq_org=30, freq_desired=200):
    """
    :param index: the index of the signal with freq=freq_org
    :param freq_org: the original freq of the signal
    :param freq_desired: the desired freq of the signal
    :return: the index of the signal with freq=freq_desired
    """
    return int(index * freq_desired / freq_org)

# class ODEParamsNumpy:
#     def __init__(self):
#         self.A = 0.005  # mV
#         self.f1 = 0.1  # mean 1
#         self.f2 = 0.25  # mean 2
#         self.c1 = 0.01  # std 1
#         self.c2 = 0.01  # std 2
#         self.rrpc = utils.generate_omega_function(self.f1, self.f2, self.c1, self.c2)
#         self.h = 1 / 216
