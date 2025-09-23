import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=1,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

    def forward(self, batch):
        return self.net(batch)

class IMUEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.imu_in_channels
        output_dim = args.imu_out_channels
        self.net = torch.nn.Sequential(
            torch.nn.GroupNorm(args.imu_group_norm, input_dim),
            Block(input_dim, args.size_embeddings, 3),
            Block(args.size_embeddings, output_dim, 3),
        )
        self.gru = nn.GRU(input_size=output_dim, hidden_size=args.size_embeddings, batch_first=True,
                          bidirectional=True)

    def forward(self, data):
        # return the last hidden state
        data = data.permute(0, 3, 1, 2) # (B, User, T, accl+gyro(6)) -> (B, accl+gyro(6), User, T)
        x = self.net(data) # (B, feature_dim, User, T)
        x = x.permute(0, 2, 3, 1).contiguous() # (B, feature_dim, User, T) -> (B, User, T, feature_dim)
        B, U, T, F = x.shape
        x = x.view(B * U, T, F)
        _, hidden = self.gru(x)
        output = hidden[0].view(B, U, -1)
        return output
        
        


class CAMEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim_kp = args.kp_in_channels  # Weighted motion keypoint features + max confidence
        output_dim = args.cam_out_channels  # Final embedding size

        self.kp_fc = nn.Sequential(
            torch.nn.GroupNorm(args.cam_group_norm, input_dim_kp),
            Block(input_dim_kp, args.size_embeddings, 3),
            Block(args.size_embeddings, output_dim, 3),
        )
        
        self.box_fc = nn.Sequential(
            Block(args.box_in_channels, args.size_embeddings, 3),
            Block(args.size_embeddings, output_dim, 3),
        )

        self.fc = nn.Linear(output_dim*2, output_dim)

        # Temporal processing with GRU
        self.gru = nn.GRU(input_size=output_dim, hidden_size=args.size_embeddings, batch_first=True,
                          bidirectional=True)

    def forward(self, box, kp):
        """
        box: (B, User, T, 5)  -> Bounding box (x, y, w, h, c)
        kp: (B, User, T, 3, 3) -> 3 keypoints (shoulder, elbow, wrist) each with (x, y, c)
        """
        B, U, T, _ = box.shape  # (Batch, Users, Time, (x, y, w, h, c))
        B, U, T, K, _ = kp.shape  # (Batch, Users, Time, 3 Keypoints, (x, y, c))

        # Compute motion vectors (Δx, Δy) between consecutive frames
        motion_box = box[..., 1:, :4] - box[..., :-1, :4]  # (B, U, T-1, 5)
        motion_kp = kp[..., 1:, :, :2] - kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)

        # Apply softmax over confidence scores across keypoints
        attention_weights_box = F.softmax(box[..., 1:, 4], dim=-1).unsqueeze(-1)  # (B, U, T-1, 1)
        attention_weights_kp = F.softmax(kp[..., 1:, :, 2], dim=-1).unsqueeze(-1)  # (B, U, T-1, 3, 1)

        # Weighted sum of motion vectors
        weighted_motion_box = motion_box * attention_weights_box  # (B, U, T-1, 4)
        weighted_motion_kp = motion_kp * attention_weights_kp  # (B, U, T-1, 3, 2)

        # Use previous frame's keypoint as reference position
        reference_position_box = box[..., :-1, :4] # (B, U, T-1, 4)
        new_box = reference_position_box + weighted_motion_box
        reference_position_kp = kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)
        new_kp = reference_position_kp + weighted_motion_kp  # (B, U, T-1, 3, 2)

        # # Concatenate new motion-based keypoint features
        kp_features = new_kp.view(B, U, T - 1, -1)  # (B, U, T-1, 6)
        box_features = new_box

        # Pass through Conv Blocks
        kp_features = kp_features.permute(0, 3, 1, 2)  # (B, U, T, 10) -> (B, 10, U, T)
        kp_features = self.kp_fc(kp_features)  # (B, U, T, output_dim)
        kp_features = kp_features.permute(0, 2, 3, 1).contiguous()  # (B, U, T, output_dim) -> (B, U, T, output_dim)
        
        box_features = box_features.permute(0, 3, 1, 2)
        box_features = self.box_fc(box_features)
        box_features = box_features.permute(0, 2, 3, 1).contiguous()
        
        kp_features = torch.cat((kp_features, box_features), dim=-1)
        kp_features = self.fc(kp_features)

        # Process with GRU
        kp_features = kp_features.view(B * U, T-1, -1)  # (B * U, T, output_dim)
        _, hidden = self.gru(kp_features)  # (1, B * U, size_embeddings)

        # Reshape back to (B, U, size_embeddings)
        output = hidden[0].view(B, U, -1)

        return output
