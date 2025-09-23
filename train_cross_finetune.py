import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from datasets.dataset_cross import AssociationDataset
from models.model import AssociationModel
from modules.loss import ContrastiveLoss

import wandb


# def add_noise(tensor, noise_level=0.02):
#     """Applies Gaussian noise to the input tensor for data augmentation."""
#     return tensor + noise_level * torch.randn_like(tensor)


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, grad_accum_steps):
    """Trains for one epoch with mixed precision and gradient accumulation."""
    model.train()
    train_loss = 0
    train_accuracy = 0
    optimizer.zero_grad()

    for i, data in enumerate(tqdm(dataloader, desc="Training")):
        box = data['box'].to(device)
        arm_right = data['arm_right'].to(device)
        imu_right = data['imu_right'].to(device)
        mask_cam = data['mask_cam'].to(device)
        mask_imu = data['mask_imu'].to(device)
        order = data['order'].to(device)

        arm_left = data['arm_left'].to(device)
        imu_left = data['imu_left'].to(device)

        arm = torch.cat((arm_right, arm_left), dim=1)
        imu = torch.cat((imu_right, imu_left), dim=1)
        mask_imu = torch.cat((mask_imu, mask_imu), dim=1)
        mask_cam = torch.cat((mask_cam, mask_cam), dim=1)
        order = torch.cat((order, order + box.size(1)), dim=1)

        with torch.amp.autocast('cuda'):  # Updated AMP syntax
            sim_matrix = model(box, arm, imu)
            loss, accuracy = criterion(sim_matrix, order, mask_cam, mask_imu)

        # Scale loss and backpropagate
        scaler.scale(loss).backward()

        if (i + 1) % grad_accum_steps == 0:  # Perform optimizer step only after accumulation
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item()
        train_accuracy += accuracy

    return train_loss / len(dataloader), train_accuracy / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the given dataset."""
    model.eval()
    test_loss = 0
    test_accuracy = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            box = data['box'].to(device)
            arm_right = data['arm_right'].to(device)
            imu_right = data['imu_right'].to(device)
            mask_cam = data['mask_cam'].to(device)
            mask_imu = data['mask_imu'].to(device)
            order = data['order'].to(device)

            arm_left = data['arm_left'].to(device)
            imu_left = data['imu_left'].to(device)

            arm = torch.cat((arm_right, arm_left), dim=1)
            imu = torch.cat((imu_right, imu_left), dim=1)
            mask_imu = torch.cat((mask_imu, mask_imu), dim=1)
            mask_cam = torch.cat((mask_cam, mask_cam), dim=1)
            order = torch.cat((order, order + box.size(1)), dim=1)

            sim_matrix = model(box, arm, imu)
            loss, accuracy = criterion(sim_matrix, order, mask_cam, mask_imu)

            test_loss += loss.item()
            test_accuracy += accuracy

    return test_loss / len(dataloader), test_accuracy / len(dataloader)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(project='association', config=args, name=f'{args.name}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Creating Data Loaders...")
    train_dataset = AssociationDataset(args, train=True)
    test_dataset = AssociationDataset(args, train=False)
    finetune_size = int(args.finetune_size * len(test_dataset))
    test_dataset, finetune_dataset = torch.utils.data.random_split(test_dataset, [len(test_dataset)-finetune_size, finetune_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True)

    print("Creating Model...")
    model = AssociationModel(args)
    model.to(device)

    print("Creating Optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler_pretrain = CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs * len(train_dataloader), eta_min=1e-6)
    lr_scheduler_finetune = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = ContrastiveLoss(args)
    scaler = torch.amp.GradScaler('cuda')  # Updated AMP syntax

    # **Pretraining Phase**
    print('Starting pretraining...')
    for epoch in range(args.pretrain_epochs):
        train_loss, train_accuracy = train_one_epoch(model, train_dataloader, optimizer, criterion, device, scaler, args.grad_accum_steps)
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)
        
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        wandb.log({'pretrain_loss': train_loss, 'pretrain_accuracy': train_accuracy, 'test_loss': test_loss, 'test_accuracy': test_accuracy})

        lr_scheduler_pretrain.step()

    print('Pretraining finished.')

    # **Fine-Tuning Phase**
    print('Starting finetuning...')
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1  # Reduce learning rate

    for epoch in range(args.finetune_epochs):
        train_loss, train_accuracy = train_one_epoch(model, finetune_dataloader, optimizer, criterion, device, scaler, args.grad_accum_steps)
        test_loss, test_accuracy = evaluate(model, test_dataloader, criterion, device)
        
        print(f"Fine-Tune Epoch {epoch+1}/{args.finetune_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        wandb.log({'finetune_loss': train_loss, 'finetune_accuracy': train_accuracy, 'test_loss': test_loss, 'test_accuracy': test_accuracy})

        lr_scheduler_finetune.step(test_loss)

    print('Finetuning finished.')


def parse_args():
    parser = argparse.ArgumentParser(description='Association Synthesized Data')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default='Association(t=60) Cross Scene Finetune', help='Experiment name')
    parser.add_argument('--data_path', type=str, default='../User Study Data/Synth/')
    parser.add_argument('--window_size', type=int, default=60, help='Window size')

    parser.add_argument('--imu_in_channels', type=int, default=6, help='Number of channels in IMU')
    parser.add_argument('--group_norm', type=int, default=2)
    parser.add_argument('--imu_out_channels', type=int, default=32, help='Number of channels in IMU')
    parser.add_argument('--kp_in_channels', type=int, default=6, help='Number of channels in keypoint')
    parser.add_argument('--cam_out_channels', type=int, default=32, help='Number of channels in CAM')
    parser.add_argument('--size_embeddings', type=int, default=128, help='Size of embeddings')
    parser.add_argument('--projection_dim', type=int, default=64, help='Projection dimension')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--finetune_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--finetune_size', type=float, default=0.05, help='Size of finetune dataset')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps')

    # Arguments for Contrastive Loss
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--reduction', default='mean', type=str)
    parser.add_argument('--negative_mode', default='unpaired', type=str)
    parser.add_argument('--symmetric_loss', default=True, type=bool)
    parser.add_argument('--learn_temperature', default=True, type=bool)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.name = f'{args.name}={args.finetune_size}'
    main(args)
