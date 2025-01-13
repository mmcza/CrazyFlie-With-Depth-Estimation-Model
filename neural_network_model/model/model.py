import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import timm


class DepthMetrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.mae  = torchmetrics.MeanAbsoluteError()

    def forward(self, preds, targets):
        preds = preds.to(targets.device)
        rmse = self.rmse(preds, targets)
        mae  = self.mae(preds, targets)

        eps = 1e-6
        ratio  = torch.max(preds / (targets + eps), targets / (preds + eps))
        delta1 = (ratio < 1.25).float().mean()
        delta2 = (ratio < 1.25**2).float().mean()
        delta3 = (ratio < 1.25**3).float().mean()

        return {
            'rmse': rmse,
            'mae': mae,
            'delta1': delta1,
            'delta2': delta2,
            'delta3': delta3
        }


class ReassembleLayer(nn.Module):
    def __init__(self, embed_dim, out_channels, scale_type='x2'):
        super().__init__()
        self.project = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        if scale_type == 'x2':
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        elif scale_type == 'x4':
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4)
        else:
            self.up = nn.Identity()


    def forward(self, x):
        B, N, D = x.shape
        h = w = int(N ** 0.5)
        if h * w != N:
            raise ValueError("Patch embeddings do not form a perfect square.")
        x = x.permute(0, 2, 1).reshape(B, D, h, w)  # [B, D, h, w]
        x = self.project(x)
        x = self.up(x)
        return x

class FusionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bicubic', align_corners=False)
        x = x + skip
        x = self.upsample(x)
        return x


class DepthEstimationDPT(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, target_size=(224, 224), vit_name='vit_base_patch16_224'):
        super().__init__()
        self.save_hyperparameters()

        self.target_size = target_size
        self.learning_rate = learning_rate


        self.metrics = DepthMetrics()

        self.criterion = nn.L1Loss()


        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)

        new_proj = nn.Conv2d(
            in_channels=2,  # Change from 3 (RGB) to 1 (grayscale)
            out_channels=self.vit.patch_embed.proj.out_channels,
            kernel_size=self.vit.patch_embed.proj.kernel_size,
            stride=self.vit.patch_embed.proj.stride,
            padding=self.vit.patch_embed.proj.padding,
            bias=self.vit.patch_embed.proj.bias is not None
        )

        new_proj.weight.data = self.vit.patch_embed.proj.weight.data.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
        self.vit.patch_embed.proj = new_proj
        # Reassemble layers
        self.reassemble1 = ReassembleLayer(embed_dim=768, out_channels=256, scale_type='x4')  # 14->56
        self.reassemble2 = ReassembleLayer(embed_dim=768, out_channels=256, scale_type='x2')  # 14->28
        self.reassemble3 = ReassembleLayer(embed_dim=768, out_channels=256, scale_type='x2')  # 14->28

        # Fusion blocks
        self.fusion1 = FusionBlock(in_channels=256)  # 28->56
        self.fusion2 = FusionBlock(in_channels=256)  # 56->112
        self.fusion3 = FusionBlock(in_channels=256)  # 112->224

        #  Head
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1)  # Additional smoothing layer
        )

    def forward(self, x):

        x = F.interpolate(x, size=self.target_size, mode='bicubic', align_corners=False)


        tokens = self.vit.forward_features(x)
        if tokens.shape[1] == 197:
            tokens = tokens[:, 1:, :]

        # Reassemble
        feat1 = self.reassemble1(tokens)
        feat2 = self.reassemble2(tokens)
        feat3 = self.reassemble3(tokens)


        feat2_up = F.interpolate(feat2, size=feat1.shape[2:], mode='bicubic', align_corners=False)
        f1 = self.fusion1(feat1, feat2_up)

        feat3_up = F.interpolate(feat3, size=f1.shape[2:], mode='bicubic', align_corners=False)
        f2 = self.fusion2(f1, feat3_up)

        zero_skip = torch.zeros_like(f2)
        f3 = self.fusion3(f2, zero_skip)

        out = self.head(f3)
        out = F.interpolate(out, size=self.target_size, mode='bicubic', align_corners=False)
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
        param_vit = []
        param_decoder = []
        for name, param in self.named_parameters():
            if name.startswith("vit."):
                param_vit.append(param)
            else:
                param_decoder.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": param_vit,     "lr": 1e-5},
                {"params": param_decoder, "lr": 1e-4},
            ],
            weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }



