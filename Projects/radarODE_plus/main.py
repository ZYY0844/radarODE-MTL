import torch, os, sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.utils import set_random_seed, set_device
from LibMTL.model import resnet_dilated
from LibMTL import Trainer
from utils.utils import shapeMetric, shapeLoss, ppiMetric, ppiLoss, anchorMetric, anchorLoss

from spectrum_dataset import dataset_concat
from nets.PPI_decoder import PPI_decoder
from nets.anchor_decoder import anchor_decoder
from nets.model import backbone, shapeDecoder
from config import prepare_args
import argparse



def parse_args(parser):
    parser.add_argument('--train_bs', default=32, type=int,
                        help='batch size for training')
    parser.add_argument('--test_bs', default=32, type=int,
                        help='batch size for test')
    parser.add_argument('--epochs', default=200,
                        type=int, help='training epochs')
    parser.add_argument('--dataset_path', default='/',
                        type=str, help='dataset path')
    # if True, only select 100 samples for training and testing
    parser.add_argument('--select_sample', default=False,
                        type=bool, help='select sample')
    parser.add_argument('--aug_snr', default=0, type=int, help='0 for no aug otherwise the SNR')
    return parser.parse_args()


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    ID_all = np.arange(1, 89)
    ID_test = np.array([75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85])
    ID_train = np.delete(ID_all, ID_test-1)

    # ID_test = np.array([1])
    # ID_train = np.array([2])
    print('ID_test', ID_test)

    radarODE_train_set = dataset_concat(
        ID_selected=ID_train, data_root=params.dataset_path)
    radarODE_test_set = dataset_concat(
        ID_selected=ID_test, data_root=params.dataset_path)

    trainloader = torch.utils.data.DataLoader(
        dataset=radarODE_train_set, batch_size=params.train_bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        dataset=radarODE_test_set, batch_size=params.test_bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # define tasks
    task_dict = {'ECG_shape': {'metrics': ['norm_MSE', 'MSE', 'CE'],
                               'metrics_fn': shapeMetric(),
                               'loss_fn': shapeLoss(),
                               'weight': [0, 0, 0]},
                 'PPI': {'metrics': ['PPI_sec', 'CE'],
                         'metrics_fn': ppiMetric(),
                         'loss_fn': ppiLoss(),
                         'weight': [0, 0]},
                 'Anchor': {'metrics': ['MSE'],
                            'metrics_fn': anchorMetric(),
                            'loss_fn': anchorLoss(),
                            'weight': [0]}}
    
    # # define backbone and en/decoders
    def encoder_class(): 
        return backbone(in_channels=50)
    num_out_channels = {'PPI': 260, 'Anchor': 200}
    decoders = nn.ModuleDict({'ECG_shape': shapeDecoder(),
                              'PPI': PPI_decoder(output_dim=num_out_channels['PPI']),
                            #   'Anchor': PPI_decoder(output_dim=num_out_channels['Anchor'])})
                              'Anchor': anchor_decoder()})

    class radarODE_plus(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class,
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, modelName, **kwargs):
            super(radarODE_plus, self).__init__(task_dict=task_dict,
                                             weighting=weighting,
                                             architecture=architecture,
                                             encoder_class=encoder_class,
                                             decoders=decoders,
                                             rep_grad=rep_grad,
                                             multi_input=multi_input,
                                             optim_param=optim_param,
                                             scheduler_param=scheduler_param,
                                             modelName=modelName,
                                             **kwargs)


    radarODE_plus_model = radarODE_plus(task_dict=task_dict,
                          weighting=params.weighting,
                          architecture=params.arch,
                          encoder_class=encoder_class,
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          modelName=params.save_name,
                          **kwargs)
    if params.mode == 'train':
        radarODE_plus_model.train(trainloader, testloader, params.epochs)
    elif params.mode == 'test':
        radarODE_plus_model.test(testloader)
    elif params.mode == 'cross_vali':
        for i in range(10):
            ID_test = np.arange(1+i*10, 11+i*10)
            ID_train = np.delete(ID_all, ID_test-1)
            params.save_name = f'{params.weighting}_cross_vali_{i}'
            radarODE_train_set = dataset_concat(
                ID_selected=ID_train, data_root=params.dataset_path)
            radarODE_test_set = dataset_concat(
                ID_selected=ID_test, data_root=params.dataset_path)

            trainloader = torch.utils.data.DataLoader(
                dataset=radarODE_train_set, batch_size=params.train_bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
            testloader = torch.utils.data.DataLoader(
                dataset=radarODE_test_set, batch_size=params.test_bs, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
            radarODE_plus_model = radarODE_plus(task_dict=task_dict,
                          weighting=params.weighting,
                          architecture=params.arch,
                          encoder_class=encoder_class,
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          save_path=params.save_path,
                          load_path=params.load_path,
                          modelName=params.save_name,
                          **kwargs)
            radarODE_plus_model.train(trainloader, testloader, params.epochs)
    else:
        raise ValueError


if __name__ == "__main__":
    n_epochs = 200
    batch_size = 22
    learning_rate = 5e-3
    lr_scheduler = 'cos'
    optimizer = 'sgd'
    weight_decay=5e-4  
    momentum=0.937
    eta_min=learning_rate * 0.01 
    T_max=100

    params = parse_args(LibMTL_args)
    params.gpu_id = '6'

    params.dataset_path = '/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_step/'
    params.save_path = '/home/zhangyuanyuan/radarODE_plus_MTL/Model_saved/'

    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    params.train_bs, params.test_bs = batch_size, batch_size
    params.epochs = n_epochs
    params.weighting = 'EGA'
    params.EGA_temp = 1
    # 100 for no noise otherwise the SNR, 6,3,0,-1,-2,-3 for SNR, 101 for 1 sec extensive abrupt noise, 111 for 1 sec mild abrupt noise
    params.aug_snr = 100 
    params.rep_grad = False
    params.multi_input = False
    params.arch = 'HPS'
    params.optim = optimizer
    params.lr, params.weight_decay, params.momentum = learning_rate, weight_decay, momentum
    params.scheduler = lr_scheduler
    params.eta_min, params.T_max = eta_min, T_max
    params.mode = 'train'
    params.save_name = f'{params.weighting}'

    main(params)
