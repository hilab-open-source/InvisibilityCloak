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
            Block(input_dim, 128, 3),
            Block(128, output_dim, 3),
            torch.nn.GroupNorm(args.group_norm, output_dim),
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
            # torch.nn.GroupNorm(args.group_norm, input_dim_kp),
            Block(input_dim_kp, 128, 3),
            Block(128, output_dim, 3),
            torch.nn.GroupNorm(args.group_norm, output_dim),
        )

        # Temporal processing with GRU
        self.gru = nn.GRU(input_size=output_dim, hidden_size=args.size_embeddings, batch_first=True,
                          bidirectional=True)

    def forward(self, kp):
        """
        box: (B, User, T, 5)  -> Bounding box (x, y, w, h, c)
        kp: (B, User, T, 3, 3) -> 3 keypoints (wrist, elbow, shoulder) each with (x, y, c)
        """
        B, U, T, K, _ = kp.shape  # (Batch, Users, Time, 3 Keypoints, (x, y, c))

        # Compute motion vectors (Δx, Δy) between consecutive frames
        motion_kp = kp[..., 1:, :, :2] - kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)

        # Apply softmax over confidence scores across keypoints
        attention_weights = F.softmax(kp[..., 1:, :, 2], dim=-1).unsqueeze(-1)  # (B, U, T-1, 3, 1)

        # Weighted sum of motion vectors
        weighted_motion = motion_kp * attention_weights  # (B, U, T-1, 3, 2)

        # Use previous frame's keypoint as reference position
        reference_position = kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)
        new_kp = reference_position + weighted_motion  # (B, U, T-1, 3, 2)

        # Concatenate new motion-based keypoint features
        kp_features = new_kp.view(B, U, T - 1, -1)  # (B, U, T-1, 6)

        # Pass through Conv Blocks
        kp_features = kp_features.permute(0, 3, 1, 2)  # (B, U, T, 6) -> (B, 6, U, T)
        kp_features = self.kp_fc(kp_features)  # (B, U, T, output_dim)
        kp_features = kp_features.permute(0, 2, 3, 1).contiguous()  # (B, U, T, output_dim) -> (B, U, T, output_dim)

        # Process with GRU
        kp_features = kp_features.view(B * U, T-1, -1)  # (B * U, T, output_dim)
        _, hidden = self.gru(kp_features)  # (1, B * U, size_embeddings)

        # Reshape back to (B, U, size_embeddings)
        output = hidden[0].view(B, U, -1)

        return output
    

class CAMEncoderRefNoAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim_kp = args.kp_in_channels  # Weighted motion keypoint features + max confidence
        output_dim = args.cam_out_channels  # Final embedding size

        self.kp_fc = nn.Sequential(
            # torch.nn.GroupNorm(args.group_norm, input_dim_kp),
            Block(input_dim_kp, 128, 3),
            Block(128, output_dim, 3),
            torch.nn.GroupNorm(args.group_norm, output_dim),
        )

        # Temporal processing with GRU
        self.gru = nn.GRU(input_size=output_dim, hidden_size=args.size_embeddings, batch_first=True)

    def forward(self, kp):
        """
        box: (B, User, T, 5)  -> Bounding box (x, y, w, h, c)
        kp: (B, User, T, 3, 3) -> 3 keypoints (wrist, elbow, shoulder) each with (x, y, c)
        """
        B, U, T, K, _ = kp.shape  # (Batch, Users, Time, 3 Keypoints, (x, y, c))

        # Compute motion vectors (Δx, Δy) between consecutive frames
        motion_kp = kp[..., 1:, :, :2] - kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)

        # Use previous frame's keypoint as reference position
        reference_position = kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)
        new_kp = reference_position + motion_kp  # (B, U, T-1, 3, 2)

        # Concatenate new motion-based keypoint features
        kp_features = new_kp.view(B, U, T - 1, -1)  # (B, U, T-1, 6)

        # Pass through Conv Blocks
        kp_features = kp_features.permute(0, 3, 1, 2)  # (B, U, T, 6) -> (B, 6, U, T)
        kp_features = self.kp_fc(kp_features)  # (B, U, T, output_dim)
        kp_features = kp_features.permute(0, 2, 3, 1).contiguous()  # (B, U, T, output_dim) -> (B, U, T, output_dim)

        # Process with GRU
        kp_features = kp_features.view(B * U, T-1, -1)  # (B * U, T, output_dim)
        _, hidden = self.gru(kp_features)  # (1, B * U, size_embeddings)

        # Reshape back to (B, U, size_embeddings)
        output = hidden[0].view(B, U, -1)

        return output


class CAMEncoderAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim_kp = args.kp_in_channels  # Weighted motion keypoint features + max confidence
        output_dim = args.cam_out_channels  # Final embedding size

        self.kp_fc = nn.Sequential(
            # torch.nn.GroupNorm(args.group_norm, input_dim_kp),
            Block(input_dim_kp, 128, 3),
            Block(128, output_dim, 3),
            torch.nn.GroupNorm(args.group_norm, output_dim),
        )

        # Temporal processing with GRU
        self.gru = nn.GRU(input_size=output_dim, hidden_size=args.size_embeddings, batch_first=True)

    def forward(self, kp):
        """
        box: (B, User, T, 5)  -> Bounding box (x, y, w, h, c)
        kp: (B, User, T, 3, 3) -> 3 keypoints (wrist, elbow, shoulder) each with (x, y, c)
        """
        B, U, T, K, _ = kp.shape  # (Batch, Users, Time, 3 Keypoints, (x, y, c))

        # Compute motion vectors (Δx, Δy) between consecutive frames
        motion_kp = kp[..., 1:, :, :2] - kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)

        # Apply softmax over confidence scores across keypoints
        attention_weights = F.softmax(kp[..., 1:, :, 2], dim=-1).unsqueeze(-1)  # (B, U, T-1, 3, 1)

        # Weighted sum of motion vectors
        weighted_motion = motion_kp * attention_weights  # (B, U, T-1, 3, 2)
        new_kp = weighted_motion  # (B, U, T-1, 3, 2)

        # Concatenate new motion-based keypoint features
        kp_features = new_kp.view(B, U, T - 1, -1)  # (B, U, T-1, 6)

        # Pass through Conv Blocks
        kp_features = kp_features.permute(0, 3, 1, 2)  # (B, U, T, 6) -> (B, 6, U, T)
        kp_features = self.kp_fc(kp_features)  # (B, U, T, output_dim)
        kp_features = kp_features.permute(0, 2, 3, 1).contiguous()  # (B, U, T, output_dim) -> (B, U, T, output_dim)

        # Process with GRU
        kp_features = kp_features.view(B * U, T-1, -1)  # (B * U, T, output_dim)
        _, hidden = self.gru(kp_features)  # (1, B * U, size_embeddings)

        # Reshape back to (B, U, size_embeddings)
        output = hidden[0].view(B, U, -1)

        return output
    




class CAMEncoderNoAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim_kp = args.kp_in_channels  # Weighted motion keypoint features + max confidence
        output_dim = args.cam_out_channels  # Final embedding size

        self.kp_fc = nn.Sequential(
            # torch.nn.GroupNorm(args.group_norm, input_dim_kp),
            Block(input_dim_kp, 128, 3),
            Block(128, output_dim, 3),
            torch.nn.GroupNorm(args.group_norm, output_dim),
        )

        # Temporal processing with GRU
        self.gru = nn.GRU(input_size=output_dim, hidden_size=args.size_embeddings, batch_first=True)

    def forward(self, kp):
        """
        box: (B, User, T, 5)  -> Bounding box (x, y, w, h, c)
        kp: (B, User, T, 3, 3) -> 3 keypoints (wrist, elbow, shoulder) each with (x, y, c)
        """
        B, U, T, K, _ = kp.shape  # (Batch, Users, Time, 3 Keypoints, (x, y, c))

        # Compute motion vectors (Δx, Δy) between consecutive frames
        motion_kp = kp[..., 1:, :, :2] - kp[..., :-1, :, :2]  # (B, U, T-1, 3, 2)
        new_kp = motion_kp

        # Concatenate new motion-based keypoint features
        kp_features = new_kp.view(B, U, T - 1, -1)  # (B, U, T-1, 6)

        # Pass through Conv Blocks
        kp_features = kp_features.permute(0, 3, 1, 2)  # (B, U, T, 6) -> (B, 6, U, T)
        kp_features = self.kp_fc(kp_features)  # (B, U, T, output_dim)
        kp_features = kp_features.permute(0, 2, 3, 1).contiguous()  # (B, U, T, output_dim) -> (B, U, T, output_dim)

        # Process with GRU
        kp_features = kp_features.view(B * U, T-1, -1)  # (B * U, T, output_dim)
        _, hidden = self.gru(kp_features)  # (1, B * U, size_embeddings)

        # Reshape back to (B, U, size_embeddings)
        output = hidden[0].view(B, U, -1)

        return output