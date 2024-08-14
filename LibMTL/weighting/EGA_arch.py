import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import control.modelsimp as msimp
from control.statesp import StateSpace
import control
import warnings
warnings.filterwarnings('ignore')
from LibMTL.weighting.abstract_weighting import AbsWeighting

def exponential_moving_average(data, alpha):
    """
    计算一维数组的指数移动平均（EMA）
    
    参数:
    data (numpy array): 输入的一维数组
    alpha (float): 平滑因子，取值范围在0到1之间
    
    返回:
    numpy array: 计算得到的EMA数组
    """
    rows, cols = data.shape
    ema = np.zeros_like(data)
    
    for row in range(rows):
        ema[row, 0] = data[row, 0]  # 初始化每行的第一个值
        for col in range(1, cols):
            ema[row, col] = alpha * data[row, col] + (1 - alpha) * ema[row, col - 1]
    return ema
# Balanced model reduction
def BTM(G):
    dim = G.size(0)
    # torch to np
    A = G.detach().cpu().numpy()
    B = np.ones(dim).reshape(-1, 1)
    C = np.ones(dim)
    D = np.array([0])

    # The full system
    fsys = StateSpace(A, B, C, D)
    n = dim
    rsys = msimp.balred(fsys, n, method='truncate')
    return torch.Tensor(rsys.A).to(G.device)

def focal_loss(loss, gamma):
    """
    计算Focal Loss
    
    参数:
    gts (torch.Tensor): 真实标签
    preds (torch.Tensor): 预测标签
    gamma (float): 调节因子
    
    返回:
    torch.Tensor: 计算得到的Focal Loss
    """
    ce_loss = loss
    pt = np.exp(-ce_loss)
    return (1 - pt) ** gamma * ce_loss

# nomalize the input data in each row
def normalize(x):
    loss_initial = x[:, 0]
    loss_diff = x[:, 1:] - x[:, :-1]
    progress = loss_diff / loss_initial[:, np.newaxis]
    return progress

class EGA(AbsWeighting):
    r"""Eccentric Gradient Alignment.
        Args:
        EGA_alpha (float, default=0.9): For calcualting the exponential moving average (Larger values prioritize more recent examples).
        EGA_gamma (float, default=1.1): For calcualting Focal loss (large for focusing on hard samples).
    """
    def __init__(self):
        super(EGA, self).__init__()
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([0.0]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        alpha, gamma = kwargs['EGA_alpha'], kwargs['EGA_gamma']
        T = 1
        # score_first_epoch = self.metrics_buffer[:, 0, 0] # [task_num, metric_num, epoch]
        # if self.epoch > 0:
        #     score = torch.Tensor(self.metrics_buffer[:, 0, self.epoch-1]).to(self.device)
        #     # set the weight for three tasks with the sum equal to 3
        #     print(score)
        #     batch_weight = self.task_num*F.softmax(score/T, dim=-1)
        # else:
        #     batch_weight = torch.ones_like(losses).to(self.device)

        if self.epoch > 1:
            w_i = torch.Tensor(self.metrics_buffer[:, 0, self.epoch-1]/self.metrics_buffer[:, 0, 1]).to(self.device)
            batch_weight = self.task_num*F.softmax(w_i/T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
            # batch_weight[0] = batch_weight[0] + self.epoch//30
        # if self.epoch >= 4:
        #     losses_epa = self.train_loss_buffer[:,:self.epoch]

        #     kpi = np.ones(len(losses)) # key performance indicator
        #     # calculate exponential moving average of the training loss
        #     losses_epa = exponential_moving_average(losses_epa, alpha)
        #     focal_losses = focal_loss(losses_epa, gamma)
        #     a=1
        # else:
        #     kpi = np.ones(len(losses))

        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        
        M = torch.matmul(grads, grads.t()) # [num_tasks, num_tasks]
        # lmbda, V = torch.symeig(M, eigenvectors=True)
        lmbda, V = torch.linalg.eigh(M, UPLO='L')

        tol = (
            torch.max(lmbda)
            * max(M.shape[-2:])
            * torch.finfo().eps
        )
        rank = sum(lmbda > tol)
        
        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]
        batch_order = torch.argsort(batch_weight, dim=-1, descending=True)
        batch_weight = batch_weight[batch_order]

        sigma = torch.diag(1 / lmbda.sqrt())
        B = lmbda[0].sqrt() * ((V @ sigma) @ V.t())

        # fact = 1 if self.epoch > 12 else 5 if self.epoch > 8 else 10
        fact = 1
        # alpha = B.sum(0)
        alpha = B.sum(0)*batch_weight*torch.Tensor([fact,1,1]).to(self.device)
        # alpha = (B.sum(0)+self.loss_scale*lmbda[0]).to(self.device)


        if self.rep_grad:
            self._backward_new_grads(alpha, per_grads=per_grads)
        else:
            self._backward_new_grads(alpha, grads=grads)
        # return alpha.detach().cpu().numpy()

        # batch_weight = torch.cat((lmbda.sqrt(), torch.tensor([1]).to(self.device)),0) if len(lmbda.sqrt()) == 2 else lmbda.sqrt()
        return alpha.detach().cpu().numpy()
