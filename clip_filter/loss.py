import torch.nn as nn
import torch

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, margin=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin    = margin
        self.eps       = eps

    def forward(self, logits, targets):
        p     = torch.sigmoid(logits)
        p_neg = torch.clamp(p - self.margin, min=0)

        loss_pos = targets       * (1 - p)    ** self.gamma_pos * torch.log(p     + self.eps)
        loss_neg = (1 - targets) * p_neg      ** self.gamma_neg * torch.log(1 - p_neg + self.eps)

        return -(loss_pos + loss_neg).mean()
