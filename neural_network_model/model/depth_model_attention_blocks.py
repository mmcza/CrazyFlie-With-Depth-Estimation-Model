import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from neural_network_model.metrics import DepthMetrics
from neural_network_model.losses import DepthLoss


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DecoderBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attention='cbam'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.attention = CBAM(out_channels) if attention == 'cbam' else None

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.attention is not None:
            x = self.attention(x)
        return self.relu(x)


class UNetWithCBAM(pl.LightningModule):
    def __init__(self, input_channels=3, learning_rate=1e-4, attention_type='cbam', target_size=(256, 256), alpha=0.85):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.target_size = target_size
        self.alpha = alpha

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder4 = DecoderBlockWithAttention(2048, 1024, attention=attention_type)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlockWithAttention(1024, 512, attention=attention_type)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlockWithAttention(512, 256, attention=attention_type)

        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder1 = DecoderBlockWithAttention(128, 64, attention=attention_type)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.metrics = DepthMetrics()
        self.loss_fn = DepthLoss(alpha=self.alpha)

    def forward(self, x):
        x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)
        x0 = self.initial(x)
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        d4 = self.upconv4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.decoder1(d1)

        out = F.interpolate(d1, size=self.target_size, mode="bilinear", align_corners=False)
        out = self.final_conv(out)
        out = torch.sigmoid(out)
        return out

    def training_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rmse", metrics["rmse"], on_epoch=True, prog_bar=True)
        self.log("train_mae", metrics["mae"], on_epoch=True, prog_bar=True)
        self.log("train_delta1", metrics["delta1"], on_epoch=True, prog_bar=True)
        self.log("train_delta2", metrics["delta2"], on_epoch=True, prog_bar=True)
        self.log("train_delta3", metrics["delta3"], on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", metrics["rmse"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae", metrics["mae"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_delta1", metrics["delta1"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_delta2", metrics["delta2"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_delta3", metrics["delta3"], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.loss_fn(preds, depths)
        metrics = self.metrics(preds, depths)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_rmse", metrics["rmse"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_mae", metrics["mae"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_delta1", metrics["delta1"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_delta2", metrics["delta2"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_delta3", metrics["delta3"], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

