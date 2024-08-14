import os, re
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset

def normal_ecg_torch_01(ECG):
    for itr in range(ECG.size(dim=0)):
        ECG[itr] = (ECG[itr]-torch.min(ECG[itr])) / \
            (torch.max(ECG[itr])-torch.min(ECG[itr]))
    return ECG
def normal_ecg(ECG):
    ECG = (ECG - np.min(ECG)) / (np.max(ECG) - np.min(ECG))
    return ECG

def normal_ecg_11(ECG):
    k = 2/(np.max(ECG)-np.min(ECG))
    ECG = -1+k*(ECG-np.min(ECG))
    return ECG

def get_all_files_in_directory(directory):
    file_paths = []
    # 使用os.walk遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for i in range(len(files) // 3):
            sst_ecg_pair = []
            sst_ecg_pair.append(os.path.join(
                root, "sst_seg_" + str(i) + '.npy'))
            sst_ecg_pair.append(os.path.join(root, "ecg_seg_"+str(i)+'.npy'))
            sst_ecg_pair.append(os.path.join(
                root, "anchor_seg_"+str(i)+'.npy'))
            file_paths.append(sst_ecg_pair)

    return file_paths

# add gaussian noise with certqin SNR to sst
def add_gaussian_sst(sst, snr_db):
    for i in range(sst.shape[0]):
        # 计算信号功率
        signal_power = np.mean(sst[i] ** 2)
        
        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # 生成高斯噪声
        noise = np.sqrt(noise_power) * np.random.randn(*sst[i].shape)
        sst[i] = sst[i] + noise
    return sst

# add abrupt noise with certqin length to sst
def add_abrupt_sst(sst, length = 1): # length 1 sec (length - 100)
    if length == 30:
        return sst
    snr_db = -9
    if length>10:
        snr_db = 0
        length -= 10
    length = int(length * 30)
    length = sst.shape[-1]-1 if length > sst.shape[-1] else length
    start = np.random.randint(0, sst.shape[-1]-length) 
    # print(sst.shape)
    for i in range(sst.shape[0]):
        # 计算信号功率
        signal_power = np.mean(sst[i][:,start:start+length] ** 2)
        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        # 生成高斯噪声
        noise = np.sqrt(noise_power) * np.random.randn(*sst[i][:,start:start+length].shape)
        sst[i][:,start:start+length] = sst[i][:,start:start+length] + noise
    return sst

def down_sample(ecg, target_len=200):
    ecg = np.interp(np.linspace(0, len(ecg), target_len),
                    np.arange(len(ecg)), ecg)
    return ecg

def des_path_finder(index,path):
    for roots, dirs, files in os.walk(path):
        for dir_ in dirs:
            if re.search(f'_{index}_', dir_):
                return os.path.join(roots, dir_)

class SpectrumECGDataset(Dataset):
    def __init__(self, sst_ecg_root, aug_snr = 100, align_length=200):
        super().__init__()

        self.sst_ecg_root = sst_ecg_root
        self.align_length = align_length
        self.aug_snr = aug_snr
        self.all_sst_ecg_files = get_all_files_in_directory(self.sst_ecg_root)
        self.index_select = np.random.choice(
                np.arange(len(self.all_sst_ecg_files)), size = int(20/100*len(self.all_sst_ecg_files)), replace=False) # used for abrupt nosing (20%)
    def __len__(self):
        return len(self.all_sst_ecg_files)

    def __getitem__(self, index):
        index = index % len(self.all_sst_ecg_files)
        sst_ecg_path = self.all_sst_ecg_files[index]
        sst_data, ecg_data, anchor_data = np.load(sst_ecg_path[0]), np.load(
            sst_ecg_path[1]), np.load(sst_ecg_path[2])
        target_len = 260
        # pad ecg_data to target length with -100
        ppi_info = np.pad(ecg_data, (0, target_len -
                          ecg_data.shape[-1]), 'constant', constant_values=-10)

        ecg_data = np.expand_dims(down_sample(ecg_data), 0)
        ppi_info = np.expand_dims(((ppi_info)), 0)
        anchor_data = np.expand_dims((anchor_data), 0)
        # sst is the normlized sst data, ppi_info is the original ecg signal with -10 padding to length of 260, ecg_data is the resampled ecg data with length of 200, anchor_data represent the position of the R peak in the ecg signal

        if self.aug_snr < 100:
            sst_data = add_gaussian_sst(sst_data, self.aug_snr)
        if self.aug_snr > 100 and index in self.index_select:
            sst_data = add_abrupt_sst(sst_data, self.aug_snr % 100)

        sst_data = torch.from_numpy(np.array(sst_data)).type(torch.FloatTensor)
        ecg_data = torch.from_numpy(np.array(ecg_data)).type(torch.FloatTensor)
        ppi_info = torch.from_numpy(np.array(ppi_info)).type(torch.FloatTensor)
        anchor_data = torch.from_numpy(np.array(anchor_data)).type(torch.FloatTensor)
        return sst_data, {'ECG_shape': ecg_data, 'PPI': ppi_info, 'Anchor': anchor_data}

def dataset_concat(ID_selected, data_root, aug_snr = 0):
    dataset = []
    for ID in ID_selected:
        ID_path = des_path_finder(ID, data_root)
        dataset = ConcatDataset([dataset, SpectrumECGDataset(sst_ecg_root=ID_path, aug_snr = aug_snr)])
    return dataset

if __name__ == '__main__':
    root = '/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_step/'
    ID = np.arange(3, 7)
    dataset = dataset_concat(ID, root, aug_snr = 101)
    print(ID, len(dataset))
    count = 0
    for item in dataset:
        print(item[0].size(), item[1]['ECG_shape'].size(), item[1]['PPI'].size(), item[1]['Anchor'].size())
        # break

