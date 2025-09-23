import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import IMUEncoder, CAMEncoder

class AssociationModel(nn.Module):
    def __init__(self, args):
        super(AssociationModel, self).__init__()
        self.imu_encoder = IMUEncoder(args)
        self.cam_encoder = CAMEncoder(args)

        # CLIP-style projection matrices (learnable parameters)
        self.imu_projection = nn.Parameter(torch.empty(args.size_embeddings, args.projection_dim))
        self.cam_projection = nn.Parameter(torch.empty(args.size_embeddings, args.projection_dim))

        # Initialize projections similar to CLIP
        nn.init.normal_(self.imu_projection, std=args.size_embeddings ** -0.5)
        nn.init.normal_(self.cam_projection, std=args.size_embeddings ** -0.5)

    def forward(self, box, kp, imu):
        '''
        box: (B, User, T, xywhc(5))
        kp: (B, User, T, joint(3), xyc(3))
        mask_cam: (B, User, T)
        imu: (B, User, T, accl+gyro(6))
        mask_imu: (B, User, T)
        '''
        feature_imu = self.imu_encoder(imu)  # (B, User, size_embeddings)
        feature_cam = self.cam_encoder(kp)  # (B, User, size_embeddings)

        # Project features into a shared space using learnable matrices
        feature_imu = feature_imu @ self.imu_projection  # (B, User, projection_dim)
        feature_cam = feature_cam @ self.cam_projection  # (B, User, projection_dim)

        # Normalize before similarity computation
        feature_imu = F.normalize(feature_imu, dim=-1)
        feature_cam = F.normalize(feature_cam, dim=-1)

        sim_matrix = torch.einsum('bud,bvd->buv', feature_imu, feature_cam)  # (B, U, U)
        return sim_matrix