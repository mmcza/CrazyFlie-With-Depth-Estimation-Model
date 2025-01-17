import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

class DepthLoss(nn.Module):
    def __init__(self, alpha=0.85):
        super(DepthLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0)
        self.compute_smoothness_loss = self._compute_smoothness_loss

    def forward(self, preds, targets):
        loss_l1 = self.l1_loss(preds, targets)
        loss_ssim = 1 - self.ssim_loss(preds, targets)
        loss_smooth = self.compute_smoothness_loss(preds)
        loss = loss_l1 * (1 - self.alpha) + loss_ssim * self.alpha + loss_smooth
        return loss

    def _compute_smoothness_loss(self, preds):
        dx = torch.abs(preds[:, :, :, :-1] - preds[:, :, :, 1:])
        dy = torch.abs(preds[:, :, :-1, :] - preds[:, :, 1:, :])
        return (dx.mean() + dy.mean()) * 0.1
