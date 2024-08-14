import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting
# only for radarODE_plus project
class Given_weight(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """

    def __init__(self):
        super(Given_weight, self).__init__()

    def backward(self, losses, **kwargs):
        # weight for ecg_shape, PPI and Anchor
        # given_weight = torch.tensor([0.0001, 1, 0.0001])
        # given_weight = torch.tensor([0.0001, 0.0001, 1])
        given_weight = torch.tensor([1, 0.0001, 0.0001])
        loss = torch.mul(losses, given_weight.to(self.device)).sum()
        loss.backward()
        return np.ones(self.task_num)
