import numpy as np
from torch.utils.data import Dataset

class AssociationDataset(Dataset):
    def __init__(self, args, train):
        super(AssociationDataset, self).__init__()
        self.data_path = args.data_path
        self.window_size = args.window_size
        if train:
            self.data = np.load(f'{self.data_path}/association_crossU{args.num_users}_train.npz')
        else:
            self.data = np.load(f'{self.data_path}/association_crossU{args.num_users}_{self.window_size}_test.npz')
        
        # if train:
        #     self.data = np.load(f'{self.data_path}/association_crossS_15_train.npz')
        # else:
        #     self.data = np.load(f'{self.data_path}/association_crossS_{self.window_size}_test.npz')

        self.order = self.data['order']
        self.box = self.data['box']
        self.arm_right = self.data['arm_right']
        self.arm_left = self.data['arm_left']
        self.mask_cam = self.data['mask_cam']
        self.imu_left = self.data['imu_left']
        self.imu_right = self.data['imu_right']
        self.mask_imu = self.data['mask_imu']
        self.scene = self.data['scene']

        self.scene_map = {
            'sports': 0,
            'independent_lab': 1,
            'classroom': 2,
            'office': 3,
            'livingroom': 4,
            'kitchen': 5,
            'diningroom': 6,
            'independent_wild': 7
        }

        print(f'Loaded data with shape: {self.box.shape}, {self.arm_right.shape}, {self.imu_right.shape}')

    def __len__(self):
        return len(self.box)
    
    def __getitem__(self, idx):
        box = self.box[idx].astype(np.float32)
        arm_right = self.arm_right[idx].astype(np.float32)
        arm_left = self.arm_left[idx].astype(np.float32)
        mask_cam = self.mask_cam[idx].astype(np.float32)
        imu_left = self.imu_left[idx].astype(np.float32)
        imu_right = self.imu_right[idx].astype(np.float32)
        mask_imu = self.mask_imu[idx].astype(np.float32)
        order = self.order[idx].astype(np.float32)
        scene = self.scene_map[self.scene[idx]]

        input = {
            'box': box,
            'arm_right': arm_right,
            'arm_left': arm_left,
            'mask_cam': mask_cam,
            'imu_left': imu_left,
            'imu_right': imu_right,
            'mask_imu': mask_imu,
            'order': order,
            'scene': scene
        }

        return input