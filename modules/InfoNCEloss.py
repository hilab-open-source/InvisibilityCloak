import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class InfoNCELoss(nn.Module):
    def __init__(self, args):
        super(InfoNCELoss, self).__init__()
        self.temperature = args.temperature

    def forward(self, sim_matrix, order, mask_cam, mask_imu):
        """
        Compute contrastive loss using InfoNCE.
        :param sim_matrix: (B, User, User) similarity matrix
        :param order: (B, User) indices of corresponding IMU features for each CAM feature
        :param mask_cam: (B, User) Binary mask indicating valid CAM users (1 for valid, 0 for invalid)
        :param mask_imu: (B, User) Binary mask indicating valid IMU users (1 for valid, 0 for invalid)
        :return: Contrastive loss scalar, accuracy
        """
        # Use the mask in the last frame
        mask_cam = mask_cam[..., -1]  # (B, User)
        mask_imu = mask_imu[..., -1]  # (B, User)

        # Apply temperature scaling
        sim_matrix /= self.temperature  # (B, User, User)
        valid_mask = (mask_cam.bool() & mask_imu.bool()).float()

        # Compute cross-entropy loss for IMU-to-CAM (rows -> columns)
        loss_imu = F.cross_entropy(sim_matrix, order.long(), reduction='mean')

        # Compute cross-entropy loss for CAM-to-IMU (columns -> rows)
        loss_cam = F.cross_entropy(sim_matrix.transpose(1, 2), order.long(), reduction='mean')
        # Symmetric InfoNCE loss
        loss = (loss_imu + loss_cam) / 2
        # loss = loss_imu

        loss = loss * valid_mask
        loss = loss.sum() / valid_mask.sum()

        matching_results = self.optimal_matching(sim_matrix)
        # Convert order to long type and compute accuracy
        correct_matches = (matching_results == order.long()) * valid_mask  # Apply mask
        accuracy = correct_matches.sum() / valid_mask.sum()  # Normalize over valid instances
        return loss, accuracy

    def optimal_matching(self, sim_matrix):
        """
        Uses Hungarian Algorithm to find the best one-to-one matches.
        :param sim_matrix: (B, User, User) similarity matrix
        :return: (B, User) matched indices
        """
        B, U, _ = sim_matrix.shape
        matched_indices = []

        for b in range(B):
            cost_matrix = -sim_matrix[b].detach().cpu().numpy()  # Convert safely to NumPy
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Solve assignment problem
            matched_indices.append(torch.tensor(col_ind, device=sim_matrix.device))  # Store matched indices

        return torch.stack(matched_indices, dim=0)  # (B, User)
