import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import random, sys, os, re, math
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from utils import global_var

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
global_var._init()
global_var.set_value('device', device)

from nets.model import ECGFormer
from data.dataloader2 import get_spectrum_ecg_trainloader, get_spectrum_ecg_testloader, get_spectrum_ecg_obj
from data.spectrum_dataset2 import normal_ecg_torch, mu_law, normal_ecg_torch_01
from utils.timestamp_create import create_folder
from utils.log_write import create_log
from nets.ODE_solver import ode1_solver

import neurokit2 as nk


train_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_new/train"
test_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_new/test"
data_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_step/"

# --------------------------------- SEED ------------------------------------- #


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(777)
# ---------------------------------------------------------------------------- #


def scale_output(input_params):
    # use input_params as the percentage range to scale the default_input
    default_input = np.array((
        ([5, 0.25, -15.0 * math.pi / 180.0,
          -100.0, 0.1, 25.0 * math.pi / 180.0,
          480.0, 0.1, 40.0 * math.pi / 180.0,
          -120, 0.1, 60.0 * math.pi / 180.0,
          8, 0.4, 135.0 * math.pi / 180.0]))).reshape((1, 15))
    default_input = torch.Tensor(default_input)
    # 2 is the factor for percentage
    input_params = (1+1*input_params) * default_input.to(device)
    # input_params = input_params.to(device)
    # input_params = default_input
    return input_params


def cross_entropy_loss_shape(ecg_rcon, ecg_gts):
    loss = nn.CrossEntropyLoss()
    ecg_rcon = ecg_rcon.squeeze(1)
    possi = ecg_gts.squeeze(1).softmax(dim=1)
    return loss(ecg_rcon, possi)


def cross_entropy_loss_ppi(ecg_rcon, ecg_gts):
    # the max/min ppi is 252/97, so we can use 155 as the range
    # count how many -1 are there in each batch of ecg_gts
    ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)
    counts = ecg_gts.size(1)-(ecg_gts == -10).sum(dim=1)
    batch_indices = torch.arange(ecg_gts.size(0))
    ecg_gts = torch.zeros_like(ecg_gts)
    ecg_gts[batch_indices, counts-1] = 1000
    loss = nn.CrossEntropyLoss()
    possi = ecg_gts.softmax(dim=1)
    return loss(ecg_rcon, possi)


def ppi_error(ecg_rcon, ecg_gts):
    ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)
    counts = ecg_gts.size(1)-(ecg_gts == -10).sum(dim=1)+1  # ppi_gts
    batch_indices = torch.arange(ecg_gts.size(0))
    ppi_pred = ecg_rcon.argmax(dim=1)
    return torch.mean(torch.abs(ppi_pred - counts)/200)


def mse_loss_ppi(ecg_rcon, ecg_gts):
    ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)
    counts = ecg_gts.size(1)-(ecg_gts == -10).sum(dim=1)
    batch_indices = torch.arange(ecg_gts.size(0))
    ecg_gts = torch.zeros_like(ecg_gts)
    ecg_gts[batch_indices, counts-1] = 1
    loss = nn.MSELoss()
    return loss(ecg_rcon, ecg_gts)


def down_sample(ecg, size_new=200):
    ecg = ecg.unsqueeze(0).unsqueeze(0)
    ecg_downsample = torch.nn.functional.interpolate(
        ecg, size=size_new,  mode='linear')
    ecg = ecg_downsample.squeeze(dim=0).squeeze(dim=0)
    return ecg


def down_sample_ode(ecg, target_len=200):
    ecg = np.interp(np.linspace(0, len(ecg), target_len),
                    np.arange(len(ecg)), ecg)
    return ecg


def train_ecg(index=1):
    n_epochs = 280*3
    batch_size = 32
    learning_rate = 5e-3
    lr_scheduler = 'cosine'
    optimizer = 'sgd'
    select_sample = False  # if True, only select 100 samples for training and testing

    # ======================================================== #
    # torch.distributed.init_process_group(backend='nccl')
    model = ECGFormer(in_channels=50).to(device)
    model._initialize_weights()

    # model=torch.nn.parallel.DistributedDataParallel(model.to(device))
    ID_all = np.arange(1, 92)
    # ID_test = np.array([71, 72, 73, 74, 75, 76, 77, 63])
    ID_test = np.arange(25, 35)
    ID_train = np.delete(ID_all, ID_test-1)
    print('ID_test', ID_test)

    trainloader = get_spectrum_ecg_obj(data_root=data_root, selected_IDs=ID_train,
                                       batch_size=batch_size, is_parallel=False, select_sample=select_sample)
    testloader = get_spectrum_ecg_obj(data_root=data_root, selected_IDs=ID_test,
                                      batch_size=batch_size, is_parallel=False, select_sample=select_sample)
    # trainloader = get_spectrum_ecg_trainloader(data_root=train_root, batch_size=batch_size, is_parallel=False, select_sample=select_sample)
    # testloader = get_spectrum_ecg_testloader(data_root=test_root, batch_size=batch_size, is_parallel=False, select_sample=select_sample)

    criterion_ppi = ppi_error  # calucate the ppi error in seconds
    # calucate the cross entropy loss for ppi as a classification problem and used for back propagation
    criterion_ce_ppi = cross_entropy_loss_ppi
    criterion_ce_shape = cross_entropy_loss_shape
    criterion_mse_shape = nn.MSELoss()

    if optimizer == 'adamw':
        optimizer = optim.AdamW(lr=learning_rate, params=model.parameters(
        ), weight_decay=5e-4)  # not very useful for this project
    else:
        optimizer = optim.SGD(
            lr=learning_rate, params=model.parameters(), weight_decay=5e-4, momentum=0.937)

    if lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, eta_min=learning_rate * 0.01, T_max=n_epochs/10)
    elif lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer, gamma=0.9, step_size=1)

    # state_dict_path="/home/zhangyuanyuan/ECGFormer_/Model_saved/best_model_endless_mse_new.pth"
    # # checkpoint = torch.load(state_dict_path)
    # model.load_state_dict(torch.load(state_dict_path))
    # # optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = 0
    # print('Load epoch {} success'.format(start_epoch))

    # ------------------------------ Start Training ---------------------------- #
    print()
    print("================= Training Configuration ===================")
    print("trainloader size:", len(trainloader) * batch_size)
    print("testloader size:", len(testloader) * batch_size)
    print("epoch:", n_epochs)
    print("batch size:", batch_size)
    print("optimizer:", optimizer)
    print("scheduler:", lr_scheduler)
    print("initial learning rate:", learning_rate)
    print("=============================================================")
    mse_loss_min = 1000000
    train_loss_array_ce, valid_loss_array_ce = [], []
    train_loss_array_mse, valid_loss_array_mse = [], []
    train_loss_array_mse_norm, valid_loss_array_mse_norm = [], []
    train_loss_array_mtl, valid_loss_array_mtl = [], []
    train_loss_array_ppi, valid_loss_array_ppi = [], []
    best_model = None
    best_model_name = None

    # suffix = f'cross_vali_{index}_mse_'
    w_shape, w_ppi = 2, 1
    suffix = f'_MTL_all_{w_shape}_{w_ppi}_test'
    # suffix = '_MTL_ppi_'
    # suffix = '_MTL_all_'
    match_shape = re.search('_shape_', suffix)
    match_mtl = re.search('_all_', suffix)
    match_ppi = re.search('_ppi_', suffix)
    print('MTL') if match_mtl else print('shape') if match_shape else print('PPI')
    log_folder = create_folder(suffix)
    for epoch in range(n_epochs):
        train_loss_ce = 0
        train_loss_mse = 0
        train_loss_norm_mse = 0
        train_loss_mtl = 0
        train_loss_ppi = 0
        train_loss_ppi_sec = 0
        train_loop = tqdm(enumerate(trainloader), total=len(trainloader))
        model.train()
        for i, (maps, ppi_labels, ecg_lables) in train_loop:
            inputs = maps.to(device)
            ppi_gts = ppi_labels.to(device)
            ecg_gts = normal_ecg_torch_01(ecg_lables).to(device)

            ppi, ecg_shape = (model(inputs))
            pred_norm = normal_ecg_torch_01(torch.clone(ecg_shape).detach())

            loss_ppi = criterion_ce_ppi((ppi), ppi_gts)
            loss_ecg_ce = criterion_ce_shape(ecg_shape, ecg_gts)
            loss_ecg_mse = criterion_mse_shape(ecg_shape, ecg_gts)
            loss_ecg_mse_norm = criterion_mse_shape(pred_norm, ecg_gts)
            loss_ppi_sec = criterion_ppi(ppi, ppi_gts)
            
            if match_mtl:
                loss_total = loss_ecg_ce + loss_ppi
            if match_shape:
                loss_total = w_shape*loss_ecg_ce + w_ppi*loss_ppi
            if match_ppi:
                loss_total = loss_ppi

            train_loss_ppi += loss_ppi.item()
            train_loss_ce += loss_ecg_ce.item()
            train_loss_mse += loss_ecg_mse.item()
            train_loss_norm_mse += loss_ecg_mse_norm.item()
            train_loss_mtl += loss_total.item()
            train_loss_ppi_sec += loss_ppi_sec.item()

            # ------------------ 清空梯度,反向传播 ----------------- #
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            # ----------------------------------------------------- #
            train_loop.set_description(f'Epoch [{epoch}/{n_epochs}]')
            train_loop.set_postfix(train_mtl_loss=loss_total.item(), train_ppi_loss=loss_ppi_sec.item(
            ), train_ce_loss=loss_ecg_ce.item(), learning_rate=optimizer.param_groups[0]['lr'])

        train_loss_array_ppi.append(train_loss_ppi/len(trainloader))
        train_loss_array_ce.append(train_loss_ce/len(trainloader))
        train_loss_array_mse.append(train_loss_mse/len(trainloader))
        train_loss_array_mse_norm.append(train_loss_norm_mse/len(trainloader))
        train_loss_array_mtl.append(train_loss_mtl/len(trainloader))

        create_log(root_path=log_folder, loss_tyoe='train_ppi',
                   loss_array=train_loss_array_ppi)
        create_log(root_path=log_folder, loss_tyoe='train_ce',
                   loss_array=train_loss_array_ce)
        create_log(root_path=log_folder, loss_tyoe='train_mse',
                   loss_array=train_loss_array_mse)
        create_log(root_path=log_folder, loss_tyoe='train_mse_norm',
                   loss_array=train_loss_array_mse_norm)
        create_log(root_path=log_folder, loss_tyoe='train_mtl',
                   loss_array=train_loss_array_mtl)
        # ------------------------------- Validation --------------------------------- #
        validation_loop = tqdm(enumerate(testloader), total=len(testloader))
        print()
        print("########################## start validation #############################")
        model.eval()
        with torch.no_grad():
            validation_loss_ce = 0
            validation_loss_mse = 0
            validation_loss_ppi = 0
            validation_loss_mtl = 0
            validation_loss_mse_norm = 0
            validation_loss_ppi_sec = 0
            for i, (maps, ppi_labels, ecg_lables) in validation_loop:
                inputs = maps.to(device)
                # inputs = normal_ecg_torch(maps).to(device)
                ecg_gts = normal_ecg_torch_01(ecg_lables).to(device)
                ppi_gts = ppi_labels.to(device)
                ppi, ecg_shape = (model(inputs))
                pred_norm = normal_ecg_torch_01(torch.clone(ecg_shape).detach())

                loss_ppi = criterion_ce_ppi((ppi), ppi_gts)
                loss_ecg_ce = criterion_ce_shape(ecg_shape, ecg_gts)
                loss_ecg_mse = criterion_mse_shape(ecg_shape, ecg_gts)
                loss_ecg_mse_norm = criterion_mse_shape(pred_norm, ecg_gts)
                loss_ppi_sec = criterion_ppi(ppi, ppi_gts)
                loss_total = w_shape*loss_ecg_ce + w_ppi*loss_ppi

                validation_loss_mse += loss_ecg_mse.item()
                validation_loss_ce += loss_ecg_ce.item()
                validation_loss_ppi += loss_ppi.item()
                validation_loss_mse_norm += loss_ecg_mse_norm.item()
                validation_loss_mtl += loss_total.item()
                validation_loss_ppi_sec += loss_ppi_sec.item()

                validation_loop.set_postfix(mtl_loss=validation_loss_mtl / (len(testloader)), test_ce_loss=validation_loss_ce / (
                    len(testloader)), test_ppi_loss=validation_loss_ppi_sec / (len(testloader)))

            # validation_selected = validation_loss_mse if match_mse else validation_loss_ce
            print('MTL') if match_mtl else print('shape') if match_shape else print('PPI')
            validation_selected = validation_loss_mtl if match_mtl else validation_loss_ce if match_shape else validation_loss_ppi
            # validation_selected = validation_loss_ce
            print(validation_selected/ (len(testloader)))
            if validation_selected < mse_loss_min:
                best_model = model
                best_model_name = 'Model_saved/best_model_' + suffix + '.pth'
                print("\033[31mbest model now:\033[0m", best_model_name, "vali_ppi=", validation_loss_ppi_sec/len(testloader),
                      "vali_mse_norm=", validation_loss_mse_norm/len(testloader), 'vali_ce=', validation_loss_ce/len(testloader),'vali_mtl=', validation_loss_mtl/len(testloader))
                torch.save(best_model.state_dict(), best_model_name)
                mse_loss_min = validation_selected
            else:
                cur_model = model
                cur_model_name = 'Model_saved/cur_model_' + suffix + '.pth'
                torch.save(cur_model.state_dict(), cur_model_name)

        valid_loss_array_mse.append(validation_loss_mse/len(testloader))
        valid_loss_array_mse_norm.append(
            validation_loss_mse_norm/len(testloader))
        valid_loss_array_ce.append(validation_loss_ce/len(testloader))
        valid_loss_array_ppi.append(validation_loss_ppi/len(testloader))
        valid_loss_array_mtl.append(validation_loss_mtl/len(testloader))

        create_log(root_path=log_folder, loss_tyoe='test_ppi',
                   loss_array=valid_loss_array_ppi)
        create_log(root_path=log_folder, loss_tyoe='test_ce',
                   loss_array=valid_loss_array_ce)
        create_log(root_path=log_folder, loss_tyoe='test_mse',
                   loss_array=valid_loss_array_mse)
        create_log(root_path=log_folder, loss_tyoe='test_mse_norm',
                   loss_array=valid_loss_array_mse_norm)
        create_log(root_path=log_folder, loss_tyoe='test_mtl',
                   loss_array=valid_loss_array_mtl)

        print()
        print("########################## end validation #############################")
        # ---------------------------------------------------------------------------- #

        scheduler.step()


def train_loop():
    for index in range(5):
        train_ecg(index=index)


if __name__ == '__main__':
    train_ecg(index=1)

    # train_loop()
