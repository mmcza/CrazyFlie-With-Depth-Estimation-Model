import torch
import torch.nn as nn
import torch.onnx
from monai.networks.nets import resnet34

#from lightningmodule import segmentationModule
from neural_network_model.model.model import DepthEstimationDPT
from neural_network_model.model.depth_model_unetresnet import DepthEstimationUNetResNet50
from neural_network_model.model.unet_pretrained_encoder import UNetWithPretrainedEncoder

def convert_checkpoint_to_onnx(checkpoint_path, onnx_output_path, input_size=(1, 3, 224, 224)):
    """
    Converts a PyTorch checkpoint to an ONNX model.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        onnx_output_path (str): Path to save the ONNX model.
        input_size (tuple): Size of the dummy input (batch_size, channels, height, width).
    """
    # Load the model from the checkpoint
    # model = CustomSegmentationModel.load_from_checkpoint(checkpoint_path)
    # model = DepthEstimationDPT.load_from_checkpoint(
    #     checkpoint_path,
    #     lr=1e-4,
    #     pretrained=True
    # )
    # model = DepthEstimationUNetResNet50.load_from_checkpoint(
    #     checkpoint_path,
    #     lr=1e-4,
    #     pretrained=True
    # )
    model = UNetWithPretrainedEncoder.load_from_checkpoint(
        checkpoint_path,
        lr=1e-4,
        input_channels=3,
        attention_type='cbam',
        target_size=(256, 256),
        pretrained=True
    )

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input with the specified size
    dummy_input = torch.randn(*input_size, device='cuda')

    # Export the model to ONNX format
    torch.onnx.export(
        model,  # Model to export
        dummy_input,  # Dummy input for the model
        onnx_output_path,  # Output path for the ONNX file
        export_params=True,  # Store the learned parameters within the model
        opset_version=15,  # Target ONNX version
        do_constant_folding=True,  # Perform constant folding optimization
        input_names=['input'],  # Name of the input tensor
        output_names=['output']  # Name of the output tensor
    )
    print(f"Model has been converted to ONNX and saved at {onnx_output_path}")


# Example usage
checkpoint_path = '../../test_model/depth-estimation-epoch=36.ckpt'
onnx_output_path = '../../test_model/depth-estimation_model_unet_with_pretrained_encoder.onnx'
convert_checkpoint_to_onnx(checkpoint_path, onnx_output_path)