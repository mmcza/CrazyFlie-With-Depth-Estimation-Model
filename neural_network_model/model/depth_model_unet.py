import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from neural_network_model.metrics import DepthMetrics
from neural_network_model.losses import DepthLoss


class DepthEstimationUNetResNet34(pl.LightningModule):
    def __init__(
            self,
            learning_rate=1e-4,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            freeze_encoder=False,
            target_size=(256, 256),
            alpha=0.85
    ):
        super(DepthEstimationUNetResNet34, self).__init__()
        self.save_hyperparameters()
        self.target_size = target_size
        self.learning_rate = learning_rate
        self.alpha = alpha

        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        if freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.upconv4 = self.pixel_shuffle_block(512, 256)
        self.decoder4 = self.decoder_block(512, 256)

        self.upconv3 = self.pixel_shuffle_block(256, 128)
        self.decoder3 = self.decoder_block(256, 128)

        self.upconv2 = self.pixel_shuffle_block(128, 64)
        self.decoder2 = self.decoder_block(128, 64)

        self.upconv1 = self.pixel_shuffle_block(64, 16)
        self.decoder1 = self.decoder_block(16 + 64, 16)

        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

        self.metrics = DepthMetrics()
        self.loss_fn = DepthLoss(alpha=self.alpha)

    def pixel_shuffle_block(self, in_channels, out_channels, upscale_factor=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def decoder_block(self, in_channels, out_channels, dropout_p=0.3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        x0 = self.initial(x)
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        d4 = self.upconv4(x4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)

        x0_up = F.interpolate(x0, size=d1.shape[2:], mode='bilinear', align_corners=False)

        d1 = torch.cat([d1, x0_up], dim=1)
        d1 = self.decoder1(d1)

        out = F.interpolate(d1, size=self.target_size, mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        out = torch.sigmoid(out)

        return out

    def training_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_rmse', metrics['rmse'], on_epoch=True, prog_bar=True)
        self.log('train_mae', metrics['mae'], on_epoch=True, prog_bar=True)
        self.log('train_ssim', metrics['ssim'], on_epoch=True, prog_bar=True)
        self.log('train_delta1', metrics['delta1'], on_epoch=True, prog_bar=True)
        self.log('train_delta2', metrics['delta2'], on_epoch=True, prog_bar=True)
        self.log('train_delta3', metrics['delta3'], on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_rmse', metrics['rmse'], on_epoch=True, prog_bar=True)
        self.log('val_mae', metrics['mae'], on_epoch=True, prog_bar=True)
        self.log('val_ssim', metrics['ssim'], on_epoch=True, prog_bar=True)
        self.log('val_delta1', metrics['delta1'], on_epoch=True, prog_bar=True)
        self.log('val_delta2', metrics['delta2'], on_epoch=True, prog_bar=True)
        self.log('val_delta3', metrics['delta3'], on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_rmse', metrics['rmse'], on_epoch=True, prog_bar=True)
        self.log('test_mae', metrics['mae'], on_epoch=True, prog_bar=True)
        self.log('test_ssim', metrics['ssim'], on_epoch=True, prog_bar=True)
        self.log('test_delta1', metrics['delta1'], on_epoch=True, prog_bar=True)
        self.log('test_delta2', metrics['delta2'], on_epoch=True, prog_bar=True)
        self.log('test_delta3', metrics['delta3'], on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
