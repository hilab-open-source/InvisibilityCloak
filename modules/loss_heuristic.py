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
        return loss, results, matching_results, mask_cam_valid


    def optimal_matching_optimized(self, sim_matrix, mask_cam, mask_imu):
        """
        Uses the Hungarian Algorithm to find one-to-one matches using only valid entries.
        If the matching cost exceeds a threshold (indicating low similarity), the match is discarded.
        Unmatched indices are filled with -1.

        This function is adapted to handle cases where the number of devices (IMU signals)
        exceeds the number of detected users (CAM). In such cases, the cost matrix is rectangular,
        and the Hungarian algorithm returns a matching covering the smaller dimension (CAM entries).
        The mapping back to original indices ensures that each valid CAM entry is assigned at most one
        IMU device, while extra IMU devices remain unmatched.

        :param sim_matrix: (B, U, U) similarity matrix.
        :param mask_cam: (B, U) binary mask for valid CAM entries.
        :param mask_imu: (B, U) binary mask for valid IMU entries.
        :return: (B, U) tensor of matched indices for CAM (or -1 for unmatched).
        """
        B, U, _ = sim_matrix.shape
        matched_indices = []
        no_match_cost = 3.0  # Threshold for rejecting a match

        for b in range(B):
            sim = sim_matrix[b]  # Shape: (U, U)
            mask_cam_b = mask_cam[b].bool()  # Shape: (U,)
            mask_imu_b = mask_imu[b].bool()  # Shape: (U,)

            # If there are no valid CAM or IMU entries, mark all as unmatched.
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

            # Convert similarity to cost matrix by negating the similarity.
            cost_matrix = -valid_sim.detach().cpu().numpy()
            # print(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Initialize the matching result with -1 (indicating no match) for all CAM entries.
            matched = torch.full((U,), -1, dtype=torch.long, device=sim_matrix.device)
            valid_cam_indices = torch.nonzero(mask_cam_b, as_tuple=False).squeeze(1)
            valid_imu_indices = torch.nonzero(mask_imu_b, as_tuple=False).squeeze(1)

            # Map matches from the reduced indices back to the original indices.
            for i in range(len(row_ind)):
                if cost_matrix[row_ind[i], col_ind[i]] > no_match_cost:
                    # Reject match if the cost is too high.
                    continue
                full_cam_index = valid_cam_indices[col_ind[i]]
                full_imu_index = valid_imu_indices[row_ind[i]]
                matched[full_cam_index] = full_imu_index
            matched_indices.append(matched)
        return torch.stack(matched_indices, dim=0)

    

    def compute_matching_metrics(self, order, matching_results, mask):
        """
        Computes accuracy, precision, recall, and F1 score for the association task.
        
        In our setting:
        - True Positives (TP): Cases where the predicted association exactly matches the ground truth.
        - False Positives (FP): Cases where a device is incorrectly associated with a user 
        (i.e., the predicted index is not equal to the ground truth, and is not -1).
        - False Negatives (FN): Cases where an association exists in the ground truth but the 
        matching result is -1 (indicating that the system missed the association).
        
        Args:
            order: (B, U) ground truth indices for each CAM device.
            matching_results: (B, U) predicted indices from the matching algorithm (-1 if unmatched).
            mask: (B, U) binary mask indicating valid CAM entries.
            
        Returns:
            A dictionary with computed metrics: accuracy, precision, recall, and F1 score.
        """
        # Convert tensors to numpy arrays for metric computation.
        order_np = order.cpu().numpy()
        matching_np = matching_results.cpu().numpy()
        mask_np = mask.cpu().numpy().astype(bool)

        # Only consider valid CAM entries.
        valid_order = order_np[mask_np]
        valid_pred = matching_np[mask_np]

        # True Positives (TP): Correct associations.
        TP = np.sum(valid_order == valid_pred)
        
        # False Positives (FP): Cases where a device is associated with a user, 
        # but the association is incorrect (predicted != ground truth and predicted != -1).
        FP = np.sum((valid_order != valid_pred) & (valid_pred != -1))
        
        # False Negatives (FN): Cases where the ground truth indicates an association 
        # (i.e., order is not -1) but the system fails to predict it (predicted is -1).
        FN = np.sum((valid_order != -1) & (valid_pred == -1))

        # Compute accuracy, precision, recall, and F1 score.
        total = len(valid_order)
        accuracy = float(TP) / total if total > 0 else 0.0
        precision = float(TP) / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = float(TP) / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }