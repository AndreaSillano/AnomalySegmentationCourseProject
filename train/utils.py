import torch
import torch.nn as nn
import torch.nn.functional as F
class LogitNormLoss(nn.Module):

    def __init__(self, weight,t=1.0):
        super(LogitNormLoss, self).__init__()
        self.t = t
        self.weight = weight

    def forward(self, logit, target):
        norms = torch.norm(logit, p=2, dim=-1, keepdim=True) + 1e-7
        output_prob =F.softmax(norms / self.t, dim=1)
        return F.cross_entropy(output_prob, target, weight = self.weight)