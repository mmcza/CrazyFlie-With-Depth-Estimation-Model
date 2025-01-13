import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
import torchmetrics

class DepthMetrics(nn.Module):
    def __init__(self):
        super(DepthMetrics, self).__init__()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mae = torchmetrics.MeanAbsoluteError()

    def forward(self, preds, targets):
        preds = preds.to(targets.device)
        rmse = self.rmse(preds, targets)
        mae = self.mae(preds, targets)

        eps = 1e-6
        ratio = torch.max(preds / (targets + eps), targets / (preds + eps))
        delta1 = (ratio < 1.25).float().mean()
        delta2 = (ratio < 1.25 ** 2).float().mean()
        delta3 = (ratio < 1.25 ** 3).float().mean()

        return {
            'rmse': rmse,
            'mae': mae,
            'delta1': delta1,
            'delta2': delta2,
            'delta3': delta3
        }

class DepthEstimationUNetResNet50(pl.LightningModule):
    def __init__(
            self,
            learning_rate=1e-4,
            encoder_name='resnet50',
            encoder_weights='imagenet',
            freeze_encoder=False,
            target_size=(256, 256)
    ):
        super(DepthEstimationUNetResNet50, self).__init__()
        self.save_hyperparameters()
        self.target_size = target_size
        self.learning_rate = learning_rate


        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False

        # Enkoder
        self.initial = nn.Sequential(
            resnet.conv1,  # [B, 64, H/2, W/2]
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # [B, 64, H/4, W/4]
        )
        self.encoder1 = resnet.layer1  # [B, 256, H/4, W/4]
        self.encoder2 = resnet.layer2  # [B, 512, H/8, W/8]
        self.encoder3 = resnet.layer3  # [B, 1024, H/16, W/16]
        self.encoder4 = resnet.layer4  # [B, 2048, H/32, W/32]

        # Dekoder
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder4 = self.decoder_block(2048, 1024)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = self.decoder_block(1024, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = self.decoder_block(512, 256)

        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder1 = self.decoder_block(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        # Metryki
        self.metrics = DepthMetrics()

        # Strata
        self.criterion = nn.L1Loss()

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        # Enkoder
        x0 = self.initial(x)       # [B, 64, H/4, W/4]
        x1 = self.encoder1(x0)     # [B, 256, H/4, W/4]
        x2 = self.encoder2(x1)     # [B, 512, H/8, W/8]
        x3 = self.encoder3(x2)     # [B, 1024, H/16, W/16]
        x4 = self.encoder4(x3)     # [B, 2048, H/32, W/32]

        # Dekoder
        d4 = self.upconv4(x4)                      # [B, 1024, H/16, W/16]
        d4 = torch.cat([d4, x3], dim=1)             # [B, 2048, H/16, W/16]
        d4 = self.decoder4(d4)                      # [B, 1024, H/16, W/16]

        d3 = self.upconv3(d4)                       # [B, 512, H/8, W/8]
        d3 = torch.cat([d3, x2], dim=1)             # [B, 1024, H/8, W/8]
        d3 = self.decoder3(d3)                      # [B, 512, H/8, W/8]

        d2 = self.upconv2(d3)                       # [B, 256, H/4, W/4]
        d2 = torch.cat([d2, x1], dim=1)             # [B, 512, H/4, W/4]
        d2 = self.decoder2(d2)                      # [B, 256, H/4, W/4]

        d1 = self.upconv1(d2)                       # [B, 64, H/2, W/2]

        x0_up = F.interpolate(x0, size=d1.shape[2:], mode='bilinear',
                              align_corners=False)  # [B, 64, H/2, W/2]

        d1 = torch.cat([d1, x0_up], dim=1)           # [B, 128, H/2, W/2]
        d1 = self.decoder1(d1)                      # [B, 64, H/2, W/2]

        out = F.interpolate(d1, size=self.target_size, mode='bilinear', align_corners=False)
        out = self.final_conv(out)                   # [B, 1, H, W]

        return out

    def training_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.criterion(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rmse', metrics['rmse'], on_epoch=True, prog_bar=True)
        self.log('train_mae', metrics['mae'], on_epoch=True, prog_bar=True)
        self.log('train_delta1', metrics['delta1'], on_epoch=True, prog_bar=True)
        self.log('train_delta2', metrics['delta2'], on_epoch=True, prog_bar=True)
        self.log('train_delta3', metrics['delta3'], on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.criterion(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_rmse', metrics['rmse'], on_epoch=True, prog_bar=True)
        self.log('val_mae', metrics['mae'], on_epoch=True, prog_bar=True)
        self.log('val_delta1', metrics['delta1'], on_epoch=True, prog_bar=True)
        self.log('val_delta2', metrics['delta2'], on_epoch=True, prog_bar=True)
        self.log('val_delta3', metrics['delta3'], on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.criterion(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_rmse', metrics['rmse'], on_epoch=True, prog_bar=True)
        self.log('test_mae', metrics['mae'], on_epoch=True, prog_bar=True)
        self.log('test_delta1', metrics['delta1'], on_epoch=True, prog_bar=True)
        self.log('test_delta2', metrics['delta2'], on_epoch=True, prog_bar=True)
        self.log('test_delta3', metrics['delta3'], on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

