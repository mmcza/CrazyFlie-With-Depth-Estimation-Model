import torch

# Load the checkpoint
ckpt = torch.load("../../test_model/depth-estimation-epoch=26.ckpt")
print(ckpt.keys())

# Check the "callbacks" key
if 'callbacks' in ckpt:
    for callback_name, callback_data in ckpt['callbacks'].items():
        print(f"Callback {callback_name}:")
        print(callback_data)

