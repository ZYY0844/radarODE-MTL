import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LibMTL.weighting.abstract_weighting import AbsWeighting

class EGA(AbsWeighting):
    r"""Eccentric Gradient Alignment.
        Args:
        EGA_temp (float, default=1.0): The softmax temperature, large value means equal weight.
    """
    def __init__(self):
        super(EGA, self).__init__()
    def init_param(self):
        self.nadir_vector = None
        self.average_loss = 0.0
        self.average_loss_count = 0
    def backward(self, losses, **kwargs):
        T = kwargs['EGA_temp']
        warmup_epoch = 4
        batch_weight = np.ones(len(losses))

        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        
        M = torch.matmul(grads, grads.t()) # [num_tasks, num_tasks]
        lmbda, V = torch.linalg.eigh(M, UPLO='L')

        tol = (
            torch.max(lmbda)
            * max(M.shape[-2:])
            * torch.finfo().eps
        )
        rank = sum(lmbda > tol)

        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]

        sigma = torch.diag(1 / lmbda.sqrt())
        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())

        if self.epoch == warmup_epoch:
            self.average_loss += losses.detach() 
            self.average_loss_count += 1
        elif self.epoch > warmup_epoch:
            if self.nadir_vector == None:
                self.nadir_vector = self.average_loss / self.average_loss_count
                print(self.nadir_vector)
            w_i = (torch.Tensor(self.train_loss_buffer[:,self.epoch-1]).to(self.device)/self.nadir_vector)
            batch_weight = self.task_num*F.softmax(w_i/T, dim=-1) # eccentric vector
            # print(batch_weight)

        alpha = B.sum(0)*torch.Tensor(batch_weight).to(self.device)

        if self.rep_grad:
            self._backward_new_grads(alpha, per_grads=per_grads)
        else:
            self._backward_new_grads(alpha, grads=grads)
        return alpha.detach().cpu().numpy()
