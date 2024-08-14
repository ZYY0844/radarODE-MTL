import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss
from copy import deepcopy

criterion_mse = nn.MSELoss()

def normal_ecg_torch_01(ECG):
    for itr in range(ECG.size(dim=0)):
        ECG[itr] = (ECG[itr]-torch.min(ECG[itr])) / \
            (torch.max(ECG[itr])-torch.min(ECG[itr]))
    return ECG

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
    # ecg_gts[batch_indices, counts-1] = 1
    loss = nn.CrossEntropyLoss()
    possi = ecg_gts.softmax(dim=1)
    return loss(ecg_rcon, possi)

def ppi_error(ecg_rcon, ecg_gts):
    ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)
    counts = ecg_gts.size(1)-(ecg_gts == -10).sum(dim=1)+1  # ppi_gts
    batch_indices = torch.arange(ecg_gts.size(0))
    ppi_pred = ecg_rcon.argmax(dim=1)
    return torch.mean(torch.abs(ppi_pred - counts)/200)

# def anchor_error(ecg_rcon, ecg_gts):
#     ecg_rcon, ecg_gts = ecg_rcon.squeeze(1), ecg_gts.squeeze(1)

def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_residual / ss_total
    return r2

# ECG shape Metric
class shapeMetric(AbsMetric):
    def __init__(self):
        super(shapeMetric, self).__init__()
        self.mse_record = []
        self.ce_record = []
        self.norm_mse_record = []
    def update_fun(self, pred, gt):
        gt = torch.clone(gt).detach()
        gt = normal_ecg_torch_01(gt).to(pred.device)
        mse = criterion_mse(pred, gt)
        pred_norm = normal_ecg_torch_01(torch.clone(pred).detach())
        self.mse_record.append(mse.item())
        self.norm_mse_record.append(criterion_mse(pred_norm, gt).item())
        self.bs.append(pred.size()[0])
        ce = cross_entropy_loss_shape(pred, gt)
        self.ce_record.append(ce.item())
    def score_fun(self):
        records = np.array(self.mse_record)
        batch_size = np.array(self.bs)
        mse = (records*batch_size).sum()/(sum(batch_size))
        records = np.array(self.ce_record)
        ce = (records*batch_size).sum()/(sum(batch_size))
        norm_mse = (np.array(self.norm_mse_record)*batch_size).sum()/(sum(batch_size))
        return [norm_mse, mse, ce]
    def reinit(self):
        self.mse_record = []
        self.ce_record = []
        self.norm_mse_record = []
        self.bs = []
# mse loss as shapeloss
class shapeLoss(AbsLoss):
    def __init__(self):
        super(shapeLoss, self).__init__()
    def compute_loss(self, pred, gt):
        gt = torch.clone(gt).detach()
        gt = normal_ecg_torch_01(gt).to(pred.device)
        return criterion_mse(pred, gt)
        # return r2_score(pred, gt)
# PPI Metric
class ppiMetric(AbsMetric):
    def __init__(self):
        super(ppiMetric, self).__init__()
        self.ce_record = []
        self.ppi_record = [] # error in seconds
    def update_fun(self, pred, gt):
        ce = cross_entropy_loss_ppi(pred, gt)
        self.ce_record.append(ce.item())
        self.bs.append(pred.size()[0])
        ppi = ppi_error(pred, gt)
        self.ppi_record.append(ppi.item())
    def score_fun(self):
        records = np.array(self.ce_record)
        batch_size = np.array(self.bs)
        ce = (records*batch_size).sum()/(sum(batch_size))
        records = np.array(self.ppi_record)
        ppi = (records*batch_size).sum()/(sum(batch_size))
        return [ppi, ce]
    def reinit(self):
        self.ce_record = []
        self.ppi_record = []
        self.bs = []

# ce loss for ppi
class ppiLoss(AbsLoss):
    def __init__(self):
        super(ppiLoss, self).__init__()
    def compute_loss(self, pred, gt):
        return cross_entropy_loss_ppi(pred, gt)
    
# anchor Metric only use mse
class anchorMetric(AbsMetric):
    def __init__(self):
        super(anchorMetric, self).__init__()
    def update_fun(self, pred, gt):
        gt = torch.clone(gt).detach()
        # gt = normal_ecg_torch_01(gt).to(pred.device)
        pred_norm = normal_ecg_torch_01(torch.clone(pred).detach())
        mse = criterion_mse(pred_norm, gt)
        self.record.append(mse.item())
        self.bs.append(pred.size()[0])
    def score_fun(self):
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return [(records*batch_size).sum()/(sum(batch_size))]
class anchorLoss(AbsLoss):
    def __init__(self):
        super(anchorLoss, self).__init__()
    def compute_loss(self, pred, gt):
        gt = torch.clone(gt).detach()
        gt = normal_ecg_torch_01(gt).to(pred.device)
        return criterion_mse(pred, gt)