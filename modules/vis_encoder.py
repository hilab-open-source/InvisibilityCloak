from encoder_kp import Block, IMUEncoder, CAMEncoder
import torch
from torchviz import make_dot
from encoder_kp import IMUEncoder  # Adjust if it's in another path

class Args:
    imu_in_channels = 6
    imu_out_channels = 32
    imu_group_norm = 2
    size_embeddings = 64

args = Args()
model = IMUEncoder(args)

# Dummy input: (B, Users, T, imu_channels)
dummy_input = torch.randn(2, 3, 10, 6)
output = model(dummy_input)

# Visualize the graph
make_dot(output, params=dict(model.named_parameters())).render("imu_encoder", format="png")






from encoder_kp import CAMEncoder

class Args:
    kp_in_channels = 6
    cam_out_channels = 32
    cam_group_norm = 2
    size_embeddings = 64
    box_in_channels = 4

args = Args()
model = CAMEncoder(args)

# Dummy inputs
B, U, T = 2, 3, 10
box = torch.randn(B, U, T, 5)
kp = torch.randn(B, U, T, 3, 3)

output = model(box, kp)

make_dot(output, params=dict(model.named_parameters())).render("cam_encoder", format="png")