import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from neural_network_model.metrics import DepthMetrics
from neural_network_model.losses import DepthLoss

class ReassembleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, upscale_factor='x2'):
        super().__init__()
        self.projection = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.upscaling = {
            'x2': nn.ConvTranspose2d(output_dim, output_dim, kernel_size=2, stride=2),
            'x4': nn.ConvTranspose2d(output_dim, output_dim, kernel_size=4, stride=4),
            'identity': nn.Identity()
        }[upscale_factor]

    def forward(self, features):
        batch_size, num_patches, channels = features.shape
        height = width = int(num_patches ** 0.5)
        features = features.permute(0, 2, 1).reshape(batch_size, channels, height, width)
        features = self.projection(features)
        features = self.upscaling(features)
        return features


class FusionBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=2, stride=2)

    def forward(self, main_feature, skip_feature):
        main_feature = F.interpolate(main_feature, size=skip_feature.shape[2:], mode='bicubic', align_corners=False)
        combined_feature = main_feature + skip_feature
        combined_feature = self.upsample(combined_feature)
        return combined_feature


class DepthEstimationDPT(pl.LightningModule):
    def __init__(self, learning_rate, target_size=(224, 224), vit_name='vit_base_patch16_224', alpha=0.85):
        super().__init__()
        self.save_hyperparameters()
        self.target_size = target_size
        self.learning_rate = learning_rate
        self.alpha = alpha

        self.loss_function = DepthLoss(alpha=self.alpha)
        self.metrics = DepthMetrics()

        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)
        self._adjust_vit()

        self.reassemble_layer1 = ReassembleLayer(input_dim=768, output_dim=256, upscale_factor='x4')
        self.reassemble_layer2 = ReassembleLayer(input_dim=768, output_dim=256, upscale_factor='x2')
        self.reassemble_layer3 = ReassembleLayer(input_dim=768, output_dim=256, upscale_factor='x2')

        self.fusion1 = FusionBlock(num_channels=256)
        self.fusion2 = FusionBlock(num_channels=256)
        self.fusion3 = FusionBlock(num_channels=256)

        self.output_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        )

    def _adjust_vit(self):
        new_proj = nn.Conv2d(
            in_channels=3,
            out_channels=self.vit.patch_embed.proj.out_channels,
            kernel_size=self.vit.patch_embed.proj.kernel_size,
            stride=self.vit.patch_embed.proj.stride,
            padding=self.vit.patch_embed.proj.padding,
            bias=self.vit.patch_embed.proj.bias is not None
        )
        new_proj.weight.data = self.vit.patch_embed.proj.weight.data.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        self.vit.patch_embed.proj = new_proj

    def forward(self, images):
        if images.shape[2:] != self.target_size:
            images = F.interpolate(images, size=self.target_size, mode='bicubic', align_corners=False)

        tokens = self.vit.forward_features(images)
        if tokens.shape[1] == 197:
            tokens = tokens[:, 1:, :]

        features1 = self.reassemble_layer1(tokens)
        features2 = self.reassemble_layer2(tokens)
        features3 = self.reassemble_layer3(tokens)

        features2_upsampled = F.interpolate(features2, size=features1.shape[2:], mode='bicubic', align_corners=False)
        fusion1_output = self.fusion1(features1, features2_upsampled)

        features3_upsampled = F.interpolate(features3, size=fusion1_output.shape[2:], mode='bicubic', align_corners=False)
        fusion2_output = self.fusion2(fusion1_output, features3_upsampled)

        zero_skip = torch.zeros_like(fusion2_output)
        final_features = self.fusion3(fusion2_output, zero_skip)

        output = self.output_head(final_features)
        output = F.interpolate(output, size=self.target_size, mode='bicubic', align_corners=False)
        return output

    def training_step(self, batch, batch_idx):
        images, ground_truth_depths = batch
        predicted_depths = self.forward(images)
        loss = self.loss_function(predicted_depths, ground_truth_depths)
        metrics = self.metrics(predicted_depths, ground_truth_depths)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict({f'train_{k}': v for k, v in metrics.items()}, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, ground_truth_depths = batch
        predicted_depths = self.forward(images)
        loss = self.loss_function(predicted_depths, ground_truth_depths)
        metrics = self.metrics(predicted_depths, ground_truth_depths)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict({f'val_{k}': v for k, v in metrics.items()}, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, ground_truth_depths = batch
        predicted_depths = self.forward(images)
        loss = self.loss_function(predicted_depths, ground_truth_depths)
        metrics = self.metrics(predicted_depths, ground_truth_depths)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict({f'test_{k}': v for k, v in metrics.items()}, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        transformer_parameters = []
        decoder_parameters = []
        for name, param in self.named_parameters():
            if name.startswith("vit."):
                transformer_parameters.append(param)
            else:
                decoder_parameters.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": transformer_parameters, "lr": 1e-5},
                {"params": decoder_parameters, "lr": self.learning_rate},
            ],
            weight_decay=1e-4
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss"
            }
        }
