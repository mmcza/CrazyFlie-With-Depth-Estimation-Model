
import torch
import torch.nn as nn
import torchmetrics


class DepthMetrics(nn.Module):
    def __init__(self):
        super(DepthMetrics, self).__init__()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mae = torchmetrics.MeanAbsoluteError()
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, preds, targets):
        preds = preds.to(targets.device)
        rmse = self.rmse(preds, targets)
        mae = self.mae(preds, targets)
        ssim = self.ssim(preds, targets)

        eps = 1e-6
        ratio = torch.max(preds / (targets + eps), targets / (preds + eps))
        delta1 = (ratio < 1.25).float().mean()
        delta2 = (ratio < 1.25 ** 2).float().mean()
        delta3 = (ratio < 1.25 ** 3).float().mean()

        return {
            'rmse': rmse,
            'mae': mae,
            'ssim': ssim,
            'delta1': delta1,
            'delta2': delta2,
            'delta3': delta3
        }
