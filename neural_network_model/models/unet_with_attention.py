
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from torchvision import models
from .attention_blocks import CBAM

class UNetWithAttention(pl.LightningModule):
    def __init__(self, input_channels=1, learning_rate=1e-4, attention_type='cbam', target_size=(256,256)):
        super(UNetWithAttention, self).__init__()
        self.learning_rate = learning_rate
        self.target_size = target_size
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        if input_channels != 3:
            old_weight = resnet.conv1.weight.data
            resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                resnet.conv1.weight[:,0:1] = old_weight.mean(dim=1, keepdim=True)

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64, 64x64
        self.layer2 = resnet.layer2  # 128, 32x32
        self.layer3 = resnet.layer3  # 256, 16x16
        self.layer4 = resnet.layer4  # 512, 8x8

        # Dekoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = DecoderBlockWithAttention(512, 256, attention=attention_type)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlockWithAttention(256, 128, attention=attention_type)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlockWithAttention(128, 64, attention=attention_type)

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = DecoderBlockWithAttention(128, 64, attention=attention_type)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        # Metryki
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)
        self.mae_metric = torchmetrics.MeanAbsoluteError()
        self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.target_size)
        # Enkkoder
        x0 = self.initial(x)   # [B,64,128,128]
        x1 = self.maxpool(x0)  # [B,64,64,64]
        x2 = self.layer1(x1)   # [B,64,64,64]
        x3 = self.layer2(x2)   # [B,128,32,32]
        x4 = self.layer3(x3)   # [B,256,16,16]
        x5 = self.layer4(x4)   # [B,512,8,8]

        # Dekoder
        d4 = self.upconv4(x5)           # [B,256,16,16]
        d4 = torch.cat([d4, x4], dim=1) # [B,512,16,16]
        d4 = self.decoder4(d4)          # [B,256,16,16]

        d3 = self.upconv3(d4)           # [B,128,32,32]
        d3 = torch.cat([d3, x3], dim=1) # [B,256,32,32]
        d3 = self.decoder3(d3)          # [B,128,32,32]

        d2 = self.upconv2(d3)           # [B,64,64,64]
        d2 = torch.cat([d2, x2], dim=1) # [B,128,64,64]
        d2 = self.decoder2(d2)          # [B,64,64,64]

        d1 = self.upconv1(d2)           # [B,64,128,128]
        d1 = torch.cat([d1, x0], dim=1) # [B,128,128,128]
        d1 = self.decoder1(d1)          # [B,64,128,128]

        out = nn.functional.interpolate(d1, size=self.target_size, mode='bilinear', align_corners=False)
        out = self.final_conv(out) # [B,1,256,256]

        return out

    def training_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.criterion(preds, depths)
        rmse = self.rmse_metric(preds, depths)
        mae = self.mae_metric(preds, depths)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rmse", rmse, prog_bar=True)
        self.log("train_mae", mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.criterion(preds, depths)
        rmse = self.rmse_metric(preds, depths)
        mae = self.mae_metric(preds, depths)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5),
            "monitor": "val_loss"
        }
        return [optimizer], [scheduler]



class DecoderBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attention='cbam'):
        super(DecoderBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if attention == 'cbam':
            self.attention = CBAM(out_channels)
        else:
            self.attention = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.attention:
            x = self.attention(x)

        x = self.relu(x)
        return x
