
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from .attention_blocks import SEBlock, CBAM


class UNetBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attention='cbam', dropout_rate=0.3):
        super(UNetBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)

        #wybór mechanizmu uwagi w celach testowych
        if attention == 'cbam':
            self.attention = CBAM(out_channels)
        elif attention == 'se':
            self.attention = SEBlock(out_channels)
        else:
            self.attention = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.dropout(x)
        if self.attention:
            x = self.attention(x)
        x = self.relu(x)
        return x


class UNetWithAttention(pl.LightningModule):
    def __init__(self, input_channels=1, learning_rate=1e-4, attention_type='cbam'):
        super(UNetWithAttention, self).__init__()
        self.learning_rate = learning_rate

        # Enkoder
        self.encoder1 = UNetBlockWithAttention(input_channels, 64, attention=attention_type)
        self.encoder2 = UNetBlockWithAttention(64, 128, attention=attention_type)
        self.encoder3 = UNetBlockWithAttention(128, 256, attention=attention_type)
        self.encoder4 = UNetBlockWithAttention(256, 512, attention=attention_type)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.middle = UNetBlockWithAttention(512, 1024, attention=attention_type)

        # Dekoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = UNetBlockWithAttention(1024, 512, attention=attention_type)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = UNetBlockWithAttention(512, 256, attention=attention_type)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = UNetBlockWithAttention(256, 128, attention=attention_type)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = UNetBlockWithAttention(128, 64, attention=attention_type)
        self.final = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Metryki
        self.rmse_metric = torchmetrics.MeanSquaredError(squared=False)
        self.mae_metric = torchmetrics.MeanAbsoluteError()

        # Strata
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # Enkoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Bottleneck
        middle = self.middle(self.pool(enc4))

        # Dekoder
        dec4 = self.upconv4(middle)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.final(dec1)
        return output

    def training_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.criterion(preds, depths)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, depths = batch
        preds = self.forward(images)
        loss = self.criterion(preds, depths)
        rmse = self.rmse_metric(preds, depths)
        mae = self.mae_metric(preds, depths)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("val_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log("val_mae", mae, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, factor=0.5
            ),
            "monitor": "val_loss"
        }
        return [optimizer], [scheduler]
