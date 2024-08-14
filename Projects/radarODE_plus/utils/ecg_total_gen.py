import torch, re, mat73, scipy
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random, sys, os
from torch.utils.data import DataLoader
import pandas as pd
sys.path.append("../..")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
from utils import global_var
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
global_var._init()
global_var.set_value('device', device)
from nets.model import ECGFormer
from data.dataloader2 import get_spectrum_ecg_trainloader, get_spectrum_ecg_testloader, get_spectrum_ecg_obj
from data.spectrum_dataset2 import SpectrumECGDataset2
from utils.timestamp_create import create_folder
from utils.log_write import create_log
import neurokit2 as nk
import matplotlib.pyplot as plt
import neurokit2 as nk
from sklearn.metrics import mean_squared_error, r2_score
from utils.train import down_sample

print(device)

# testset = SpectrumECGDataset(sst_root=sst_root, ecg_root=ecg_root, peak_root=peak_root)
def norm_ecg(ECG):
    ECG = (ECG - np.min(ECG)) / (np.max(ECG) - np.min(ECG))
    return ECG
def norm_ecg_11(ECG):
    k=2/(np.max(ECG)-np.min(ECG))
    ECG = -1+k*(ECG-np.min(ECG))
    return ECG

# normlizaiton to [0,1]
def sst_norm_01(sst):
    for i in range(len(sst)):
        k=1/(np.max(sst[i])-np.min(sst[i]))
        sst[i] = 0+k*(sst[i]-np.min(sst[i]))
    return sst
def resample(ecg, target_len=200):
    # using numpy
    ecg=np.interp(np.linspace(0, len(ecg), target_len), np.arange(len(ecg)), ecg)
    return ecg
def load_sst(sst_path):
    """
    load sst result generated from matlab
    """
    sst_plot = mat73.loadmat(sst_path, use_attrdict=True)
    # sst_plot = scipy.io.loadmat(sst_path, use_attrdict=True)
    sst_plot = np.flip(sst_plot['SST'], 1)
    return sst_plot
def mat2df():
    """
    Read .mat file from original MMECG dataset to a pandas dataframe.
    """
    # ID is the index of data files, Obj_ID is the index for actual person under test
    columnnames = ['ID', 'Obj_ID', 'RCG', 'ECG', 'posXYZ',
                   'gender', 'age', 'physistatus']
    df = pd.DataFrame(columns=columnnames)
    df.loc[0, columnnames] = [1, 2, 3, 4, 5, 6, 7, 8]

    for ID in range(1, 92):
        data = scipy.io.loadmat(data_org_root+str(ID)+'.mat')
        Obj_ID = data['data'][0]['id'].squeeze()
        RCG = np.array([i for i in data['data'][0]['RCG']]
                       ).squeeze().transpose()
        ECG = np.array([i for i in data['data'][0]['ECG']]).squeeze()
        posXYZ = np.array([i for i in data['data'][0]['posXYZ']]).squeeze()
        gender = data['data'][0]['gender'].squeeze()
        age = data['data'][0]['age'].squeeze()
        physistatus = data['data'][0]['physistatus'].squeeze()

        df.loc[ID-1, columnnames] = [ID, int(Obj_ID), RCG,
                                     ECG, posXYZ, gender, int(age), physistatus]
    return df
def smooth2nd(x,M): ##x 为一维数组
    K = round(M/2-0.1) ##M应为奇数，如果是偶数，则取大1的奇数
    lenX = len(x)
    if lenX<2*K+1:
        print('数据长度小于平滑点数')
    else:
        y = np.zeros(lenX)
        for NN in range(0,lenX,1):
            startInd = max([0,NN-K])
            endInd = min(NN+K+1,lenX)
            y[NN] = np.mean(x[startInd:endInd])
##    y[0]=x[0]       #首部保持一致
##    y[-1]=x[-1]     #尾部也保持一致
    return(y)
def index_convert(index, freq_org=200, freq_desired=30):
    """
    :param index: the index of the signal with freq=freq_org
    :param freq_org: the original freq of the signal
    :param freq_desired: the desired freq of the signal
    :return: the index of the signal with freq=freq_desired
    """
    return int(index * freq_desired / freq_org)
def des_path_finder(index,path):
    for roots, dirs, files in os.walk(path):
        for dir_ in dirs:
            if re.search(f'_{index}_', dir_):
                return roots, dir_
def sst_finder(index, path):
    for file in os.listdir(path):
        if re.search(f'_{index}_', file):
            return file
if __name__ == "__main__":
    ecg_pattern = 'ecg_seg_'
    sst_pattern = 'sst_seg_'
    data_org_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_org/"
    train_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_new/train"
    test_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_new/test"
    # rcg_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_rcg/test"
    sst_root = '/home/zhangyuanyuan/Dataset/data_MMECG/30Hz_half/'
    interval_root = '/home/zhangyuanyuan/Dataset/data_MMECG/interval_pred/'
    state_dict_path="/home/zhangyuanyuan/ECGFormer_/Model_saved/best_model_endless_mse_new.pth"
    save_path = '/home/zhangyuanyuan/Dataset/data_MMECG/ecg_total_recover/'
    ecg_seg_index_root = '/home/zhangyuanyuan/Dataset/data_MMECG/ecg_seg_index/'
    model = ECGFormer(in_channels=50).to(device)
    model._initialize_weights()
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()
    df = mat2df()
    directory = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_new/"
    cor_final = np.array([])
    rmse_final = np.array([])
    for obj_index in range(1, 92):

        print(obj_index)
        cur_root, cur_dir = des_path_finder(obj_index, directory)
        obj_path=os.path.join(cur_root, cur_dir)

        seg_df = pd.read_csv(os.path.join(ecg_seg_index_root,f'ecg_seg_index_{obj_index}.csv'), index_col=None)
        offset = seg_df.iloc[0,0]

        ecgs = []
        ecg_orgs = []
        ssts = []
        file_num = len(os.listdir(obj_path))//2
        interval_gts = np.load(os.path.join(interval_root, f'{obj_index}_gts.npy'))
        interval_pred = np.load(os.path.join(interval_root, f'{obj_index}_kde_nosample.npy')).astype(int)
       
        for files in range(file_num):
            temp_ecg = np.load(os.path.join(obj_path, f'{ecg_pattern}{files}.npy'))
            ecg_orgs.append(temp_ecg)
            temp_ecg = resample(temp_ecg, 200)
            ecgs.append(temp_ecg)
            temp_sst = np.load(os.path.join(obj_path, f'{sst_pattern}{files}.npy'))
            ssts.append(temp_sst)

        ecg_tensor=torch.from_numpy(np.array(ecgs)).float()
        sst_tensor=torch.from_numpy(np.array(ssts)).float()
        dataset = TensorDataset(sst_tensor, ecg_tensor)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        valid_prediction=[]
        ECG_gts=[]
        ECG_pred=[]
        rcgs=[]
        ssts=[]
        with torch.no_grad():
            validation_loop = tqdm(enumerate(dataloader), total=len(dataloader)) 
            for i, (maps, labels) in validation_loop:
                inputs = maps.to(device)
                gts = labels.to(device)
                predictions = model(inputs)
                ECG_gts.append(labels)
                ECG_pred.append(predictions)
                ssts.append(inputs)

        # calculate rmse error for data
        rmses = np.array([])
        mses = np.array([])
        cors = np.array([])
        for i in range(0, len(ECG_gts)):
            ecg_gts=ECG_gts[i].detach().cpu().numpy().squeeze()
            ecg_gts=np.interp(np.linspace(0, len(ecg_gts), 200), np.arange(len(ecg_gts)), ecg_gts)
            ecg_pred=ECG_pred[i].detach().cpu().numpy().squeeze()[:200]
            factor = 1/(np.max(ecg_gts)-np.min(ecg_gts))
            ecg_min = np.min(ecg_gts)
            ecg_gts = norm_ecg(ecg_gts)
            ecg_pred = norm_ecg(ecg_pred)
            mse=mean_squared_error(ecg_gts/factor, ecg_pred/factor)
            mses = np.append(mses, (mse))
            rmses = np.append(rmses, np.sqrt(mse))
            cor = np.corrcoef(ecg_gts, ecg_pred)[0,1]
            cors = np.append(cors, cor)
        print("RMSE Mean:",np.mean(rmses),"Median:",np.median(rmses), "Cor", np.mean(cors))
        cor_final = np.append(cor_final, np.mean(cors))
        rmse_final = np.append(rmse_final, np.mean(rmses))
        # concatenate all the ecg
        ecg_total = np.array([])
        ecg_total_gts = np.array([])
        # np.save(os.path.join(save_path, f'{obj_index}_pred_secg.npy'), ECG_pred.detach().cpu().numpy().squeeze().squeeze())
        # np.save(os.path.join(save_path, f'{obj_index}_gts_secg.npy'),ecgs)
        selected_interval = interval_gts
        for i in range(len(selected_interval)):
            ecg = ecg_orgs[i]
            ecg_total_gts = np.concatenate((ecg_total_gts, ecg))
            cur_ecg = resample(ECG_pred[i].detach().cpu().numpy().squeeze().squeeze(), selected_interval[i])
            ecg_total =  np.concatenate((ecg_total, cur_ecg))
        # append at the beginning with length offset
        ecg_total = np.concatenate((np.ones(offset)*ecg_total[0], ecg_total)) if offset>0 else np.delete(ecg_total, np.arange(-offset))
        print(len(ecg_total))
        if len(ecg_total) >= 35505:
            ecg_total = ecg_total[:35505]
        else:
            ecg_total = np.concatenate((ecg_total, np.ones(35505-len(ecg_total))*ecg_total[-1]))
        print(len(ecg_total))
        # np.save(os.path.join(save_path, f'{obj_index}_pred_perfect.npy'), ecg_total)
        # np.save(os.path.join(save_path, f'{obj_index}_gts.npy'), ecg_total_gts)
    print(np.mean(cor_final))
    print(np.mean(rmse_final))
    np.save(os.path.join(save_path, f'cor_final.npy'), cor_final)
    np.save(os.path.join(save_path, f'rmse_final.npy'), rmse_final)
        









#         sst_path = os.path.join(sst_root, sst_finder(obj_index, sst_root))

    #     sst_obj = load_sst(os.path.join(sst_root, sst_path))
    #     print(len(interval))
    #     sst_dataset = []
    #     sst_length = sst_obj.shape[2]
    #     cur_seg_index = 0
    #     input_len = 118
    #     total_seg_count = len(interval)
    #     for i in range(total_seg_count):
    #         cur_seg_len = index_convert(interval[i])
    #         padding_len = (input_len - cur_seg_len)//2
    #         cur_seg_range = np.arange(cur_seg_index-padding_len, cur_seg_index+cur_seg_len+padding_len)
    #         if len(cur_seg_range) < input_len:
    #             cur_seg_range = np.arange(cur_seg_index-padding_len, cur_seg_index+cur_seg_len+padding_len+1)

    #         if cur_seg_range[-1]<sst_length:
    #             if cur_seg_range[0]>=0:
    #                 sst_seg = sst_norm_01(deepcopy(sst_obj[:, :, cur_seg_range]))
    #                 sst_dataset.append(sst_seg)
    #             else:
    #                 sst_seg = sst_norm_01(deepcopy(sst_obj[:, :, :input_len]))
    #                 sst_dataset.append(sst_seg)
    #         else:
    #             sst_seg = sst_norm_01(deepcopy(sst_obj[:, :, -input_len:]))
    #             sst_dataset.append(sst_seg)

    #         cur_seg_index += cur_seg_len
    #         # if cur_seg_index >= sst_length:
    #         #     break

    #     data_tensor=torch.from_numpy(np.array(sst_dataset)).float()
    #     dataset = TensorDataset(data_tensor,data_tensor)
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    #     valid_prediction=[]
    #     # ECG_gts=[]
    #     ECG_pred=[]
    #     rcgs=[]
    #     # ssts=[]
    #     with torch.no_grad():
    #         validation_loop = tqdm(enumerate(dataloader), total=len(dataloader)) 
    #         for i, (maps, _) in validation_loop:
    #             inputs = maps.to(device)
    #             predictions = model(inputs)
    #             ECG_pred.append(predictions)
    #             # ssts.append(inputs)

    #     total_length = np.sum(interval)
    #     ecg_total = np.array([])
    #     for i in range(len(interval)):
    #         cur_seg_len = interval[i]
    #         cur_ecg = down_sample(ECG_pred[i].detach().cpu().numpy().squeeze().squeeze(), int(cur_seg_len))
    #         ecg_total =  np.concatenate((ecg_total, cur_ecg))

    #     ecg_total_gts =  df.loc[obj_index-1, 'ECG']
    #     print(len(ecg_total))
    #     np.save(os.path.join(save_path, f'{obj_index}_pred.npy'), ecg_total)






        # sst_obj = load_sst(os.path.join(sst_root, sst_path))
        # pred_obj_median = np.load(interval_path_median)
        # pred_obj_median_smooth = smooth2nd(pred_obj_median, 3000)

        # pred_obj_cur = pred_obj_median_smooth
        # seg_mean = np.mean(pred_obj_cur)
        # seg_length = [] # the length for each segment
        # flag = 35505
        # cur_seg_index = 0
        # total_seg_count = 0
        # while True:
        #     cur_seg_length = int(pred_obj_cur[cur_seg_index])
        #     seg_length.append(cur_seg_length)
        #     cur_seg_index += cur_seg_length
        #     total_seg_count += 1
        #     flag = flag - cur_seg_length
        #     if flag <= int(pred_obj_cur[cur_seg_index]):
        #         seg_length.append(flag)
        #         total_seg_count += 1
        #         break   
        # sst_dataset = []
        # sst_length = sst_obj.shape[2]
        # cur_seg_index = 0
        # input_len = 118

        # for i in range(total_seg_count):
        #     cur_seg_len = index_convert(seg_length[i])
        #     padding_len = (input_len - cur_seg_len)//2
        #     cur_seg_range = np.arange(cur_seg_index-padding_len, cur_seg_index+cur_seg_len+padding_len)
        #     if len(cur_seg_range) < input_len:
        #         cur_seg_range = np.arange(cur_seg_index-padding_len, cur_seg_index+cur_seg_len+padding_len+1)
        #     sst_dataset.append(sst_norm_01(sst_obj[:, :, cur_seg_range]))
        #     cur_seg_index += cur_seg_len
        #     if cur_seg_index >= sst_length:
        #         break
        # data_tensor=torch.from_numpy(np.array(sst_dataset)).float()
        # dataset = TensorDataset(data_tensor,data_tensor)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # valid_prediction=[]
        # # ECG_gts=[]
        # ECG_pred=[]
        # rcgs=[]
        # ssts=[]
        # with torch.no_grad():
        #     validation_loop = tqdm(enumerate(dataloader), total=len(dataloader)) 
        #     for i, (maps, _) in validation_loop:
        #         inputs = maps.to(device)
        #         predictions = model(inputs)
        #         ECG_pred.append(predictions)
        #         ssts.append(inputs)
        # total_length = np.sum(seg_length)
        # ecg_total = np.array([])
        # for i in range(total_seg_count):
        #     cur_seg_len = seg_length[i]
        #     cur_ecg = down_sample(ECG_pred[i].detach().cpu().numpy().squeeze().squeeze(), cur_seg_len)
        #     ecg_total =  np.concatenate((ecg_total, cur_ecg))
        # np.save(os.path.join(save_path, f'{obj_index}.npy'), ecg_total)
        # if len(ecg_total) <35505:
        #     print("error in ", obj_index)