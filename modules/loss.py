import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc


class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        """
        Contrastive loss for IMU-CAM feature matching using InfoNCE.
        :param temperature: Scaling factor for similarity scores.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = args.temperature
        self.symmetric_loss = args.symmetric_loss

    def forward(self, sim_matrix, order, mask_cam, mask_imu):
        """
        Compute contrastive loss using InfoNCE.
        :param sim_matrix: (B, User, User) similarity matrix
        :param order: (B, User) indices of corresponding IMU features for each CAM feature
        :param mask_cam: (B, User) Binary mask indicating valid CAM users (1 for valid, 0 for invalid)
        :param mask_imu: (B, User) Binary mask indicating valid IMU users (1 for valid, 0 for invalid)
        :return: Contrastive loss scalar, accuracy, precision, recall, F1 score, AUC, matching results
        """
        # Use the mask from the last frame
        mask_cam_valid = mask_cam[..., -1].float()  # shape: (B, User)
        mask_imu_valid = mask_imu[..., -1].float()    # shape: (B, User)

        # Apply temperature scaling
        sim_matrix /= self.temperature  # (B, User, User)

        # Compute InfoNCE loss using valid masks that match the loss shape (B, User)
        if self.symmetric_loss:
            # For the IMU->CAM direction, use mask_imu_valid to mask the loss per row.
            loss_imu = F.cross_entropy(sim_matrix, order.long(), reduction='none')  # shape: (B, User)
            loss_imu = loss_imu * mask_imu_valid
            loss_imu = loss_imu.sum() / mask_imu_valid.sum()

            # For the CAM->IMU direction, use mask_cam_valid to mask the loss per row.
            loss_cam = F.cross_entropy(sim_matrix.transpose(1, 2), order.long(), reduction='none')  # shape: (B, User)
            loss_cam = loss_cam * mask_cam_valid
            loss_cam = loss_cam.sum() / mask_cam_valid.sum()

            loss = (loss_imu + loss_cam) / 2  # Symmetric InfoNCE loss
        else:
            # For non-symmetric loss, use mask_cam_valid (assuming CAM determines validity)
            loss = F.cross_entropy(sim_matrix.transpose(1, 2), order.long(), reduction='none')  # shape: (B, User)
            loss = loss * mask_cam_valid
            loss = loss.sum() / mask_cam_valid.sum()  # Normalize over valid instances

        # Compute optimal matching (e.g., via the Hungarian algorithm)
        matching_results = self.optimal_matching_optimized(sim_matrix, mask_cam_valid, mask_imu_valid)

        # Compute accuracy only on valid CAM samples
        results = self.compute_matching_metrics(order, matching_results, mask_cam_valid)
        # accuracy = results['accuracy']
        return loss, results, matching_results


    def optimal_matching_optimized(self, sim_matrix, mask_cam, mask_imu):
        """
        Uses the Hungarian Algorithm to find one-to-one matches using only valid entries.
        Unmatched indices are filled with -1.

        :param sim_matrix: (B, U, U) similarity matrix.
        :param mask_cam: (B, U) binary mask for valid CAM entries.
        :param mask_imu: (B, U) binary mask for valid IMU entries.
        :return: (B, U) tensor of matched indices for CAM (or -1 for unmatched).
        """
        B, U, _ = sim_matrix.shape
        matched_indices = []

        for b in range(B):
            sim = sim_matrix[b]  # (U, U)
            mask_cam_b = mask_cam[b].bool()  # (U,)
            mask_imu_b = mask_imu[b].bool()  # (U,)

            # If no valid CAM or IMU entries, mark all as unmatched.
            if mask_cam_b.sum() == 0 or mask_imu_b.sum() == 0:
                matched = torch.full((U,), -1, dtype=torch.long, device=sim_matrix.device)
                matched_indices.append(matched)
                continue

            # Extract submatrix of valid entries.
            valid_sim = sim[mask_imu_b][:, mask_cam_b]
            if valid_sim.numel() == 0:
                matched = torch.full((U,), -1, dtype=torch.long, device=sim_matrix.device)
                matched_indices.append(matched)
                continue
            
            # Convert similarity to cost matrix (negative similarity).
            cost_matrix = -valid_sim.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Initialize the full matching result with -1.
            matched = torch.full((U,), -1, dtype=torch.long, device=sim_matrix.device)
            valid_cam_indices = torch.nonzero(mask_cam_b, as_tuple=False).squeeze(1)
            valid_imu_indices = torch.nonzero(mask_imu_b, as_tuple=False).squeeze(1)

            # Map matches from the reduced indices back to the original indices.
            for i in range(len(row_ind)):
                full_cam_index = valid_cam_indices[col_ind[i]]
                full_imu_index = valid_imu_indices[row_ind[i]]
                matched[full_cam_index] = full_imu_index
            matched_indices.append(matched)
        #  Under the CAM, which IMU is matched with which CAM
        #  If there is no match, it will be -1
        #  If there is a match, it will be the index of the IMU
        #  For example, if CAM 0 matches with IMU 2, then matched[0] = 2
        #  If CAM 1 does not match with any IMU, then matched[1] = -1
        return torch.stack(matched_indices, dim=0)

    

    def compute_matching_metrics(self, order, matching_results, mask):
        """
        Computes accuracy, precision, recall, F1 score, and ROC curve metrics for matching.

        :param order: (B, U) ground truth indices for each CAM device.
        :param matching_results: (B, U) predicted indices from the matching algorithm (-1 if unmatched).
        :param mask: (B, U) binary mask indicating valid CAM entries.
        :param sim_matrix: (B, U, U) similarity scores used for matching.
        :return: Dictionary with computed metrics.
        """
        order_np = order.cpu().numpy()
        matching_np = matching_results.cpu().numpy()
        mask_np = mask.cpu().numpy().astype(bool)

        # Only consider valid CAM entries.
        valid_order = order_np[mask_np]
        valid_pred = matching_np[mask_np]

        # Compute true positives, false positives, and false negatives.
        tp = np.sum(valid_order == valid_pred)
        fp = np.sum((valid_order != valid_pred) & (valid_pred != -1))
        fn = np.sum((valid_order != -1) & (valid_pred == -1))

        accuracy = float(tp) / len(valid_order) if len(valid_order) > 0 else 0.0
        precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }



    def optimal_matching(self, sim_matrix, mask_cam, mask_imu):
        """
        Uses Hungarian Algorithm to find the best one-to-one matches.
        :param sim_matrix: (B, User, User) similarity matrix
        :return: (B, User) matched indices
        """
        B, U, _ = sim_matrix.shape
        matched_indices = []

        for b in range(B):
            import pdb; pdb.set_trace()
            cost_matrix = -sim_matrix[b].detach().cpu().numpy()  # Convert safely to NumPy
            row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Solve assignment problem
            matched_indices.append(torch.tensor(col_ind, device=sim_matrix.device))  # Store matched indices

        return torch.stack(matched_indices, dim=0)  # (B, User)
