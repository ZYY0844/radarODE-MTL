import torch, sys, os, re
import numpy as np
# from data.spectrum_dataset import SpectrumECGDataset
BASE_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from spectrum_dataset import SpectrumECGDataset
from torch.utils.data import DataLoader, ConcatDataset

def des_path_finder(index,path):
    for roots, dirs, files in os.walk(path):
        for dir_ in dirs:
            if re.search(f'_{index}_', dir_):
                return os.path.join(roots, dir_)

def get_spectrum_ecg_obj(data_root, selected_IDs, batch_size=4, n_workers=8, is_parallel=False, select_sample=False, visual=False):
    dataset = []
    for ID in selected_IDs:
        ID_path = des_path_finder(ID, data_root)
        dataset = ConcatDataset([dataset, SpectrumECGDataset(sst_ecg_root=ID_path)])
    if select_sample:
        dataset = torch.utils.data.Subset(dataset, np.arange(0, 100))
    if is_parallel:
        ngpus_per_node = torch.cuda.device_count()
        # ngpus_per_node = None
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, )
        batch_size = batch_size // ngpus_per_node
        shuffle = False

    else:
        test_sampler = None
        shuffle = True
    if visual:
        # used for visualization
        shuffle = False
    gen = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=n_workers, pin_memory=True,
                     drop_last=True, collate_fn=dataset_collate, sampler=test_sampler)
    return gen

# DataLoader中collate_fn使用
def dataset_collate(batch):
    ssts = []
    ppis = []
    ecgs = []
    anchors = []
    for sst, ppi, ecg, anchor in batch:
        try:
            ssts.append(sst)
            ppis.append(ppi)
            ecgs.append(ecg)
            anchors.append(anchor)
        except:
            print(sst.shape)
            print(ppi.shape)
            print(ecg.shape)
            print(anchor.shape)

    ssts = torch.from_numpy(np.array(ssts)).type(torch.FloatTensor)
    ppis = torch.from_numpy(np.array(ppis)).type(torch.FloatTensor)
    ecgs = torch.from_numpy(np.array(ecgs)).type(torch.FloatTensor)
    anchors = torch.from_numpy(np.array(anchors)).type(torch.FloatTensor)
    return ssts, ppis, ecgs, anchors


if __name__ == '__main__':
    sst_ecg_root = "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_new/train/obj1_NB_2_/"
    dataroot= "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_new/"
    dataroot= "/home/zhangyuanyuan/Dataset/data_MMECG/data_seg_sce/"
    ID = np.arange(1, 3)

    batch_size = 16

    # trainloader = get_spectrum_ecg_trainloader(data_root=sst_ecg_root, batch_size=batch_size)
    trainloader = get_spectrum_ecg_obj(data_root=dataroot, selected_IDs=ID, batch_size=batch_size)

    batch_example = next(iter(trainloader))

    print("batch size:", batch_size)

    print("sst shape:", batch_example[0].shape)
    print("ppi shape:", batch_example[1].shape)
    print("ecg shape:", batch_example[2].shape)
    print("anchor shape:", batch_example[3].shape)
    # trainloader = get_spectrum_ecg_trainloader