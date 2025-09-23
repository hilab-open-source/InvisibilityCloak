import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from datasets.dataset import AssociationDataset
from models.model_input import AssociationModel
from modules.loss_heuristic import ContrastiveLoss

import wandb


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.init(project='association', 
               config=args,
               name=f'{args.name}')

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Creating Data Loaders...")
    train_dataset = AssociationDataset(args, train=True)
    test_dataset = AssociationDataset(args, train=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("Creating Model...")
    model = AssociationModel(args)
    model.to(device)

    print("Creating Optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*int(len(train_dataloader)), eta_min=1e-10)
    criterion = ContrastiveLoss(args)

    print('Starting training...')
    best_test_accuracy, best_test_f1, best_test_precision, best_test_recall = 0, 0, 0, 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_accuracy, train_precision, train_recall, train_f1 = 0, 0, 0, 0
        for i, data in enumerate(tqdm(train_dataloader)):
            box = data['box'].to(device)  # (B, User, T, xywhc(5))
            arm_right = data['arm_right'].to(device)  # (B, User, T, joint(3), xyc(3))
            mask_cam = data['mask_cam'].to(device)  # (B, User, T)
            imu_right = data['imu_right'].to(device)  #(B, User, T, accl+gyro(6))
            mask_imu = data['mask_imu'].to(device)  # (B, User, T)
            order = data['order'].to(device)  # (B, User)

            arm_left = data['arm_left'].to(device)
            imu_left = data['imu_left'].to(device)

            if args.hands == 'right':
                sim_matrix = model(box, arm_right, imu_right)
            else:
                sim_matrix = model(box, arm_left, imu_left)
            loss, results, match = criterion(sim_matrix, order, mask_cam, mask_imu)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += results['accuracy']
            train_precision += results['precision']
            train_recall += results['recall']
            train_f1 += results['f1']
            lr_scheduler.step()
        print(f'Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss/len(train_dataloader)}')
        
        model.eval()
        test_loss = 0
        test_accuracy, test_precision, test_recall, test_f1 = 0, 0, 0, 0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                box = data['box'].to(device)
                arm_right = data['arm_right'].to(device)
                mask_cam = data['mask_cam'].to(device)
                imu_right = data['imu_right'].to(device)
                mask_imu = data['mask_imu'].to(device)
                order = data['order'].to(device)

                arm_left = data['arm_left'].to(device)
                imu_left = data['imu_left'].to(device)

                if args.hands == 'right':
                    sim_matrix = model(box, arm_right, imu_right)
                else:
                    sim_matrix = model(box, arm_left, imu_left)
                loss, results, match = criterion(sim_matrix, order, mask_cam, mask_imu)
                test_loss += loss.item()
                test_accuracy += results['accuracy']
                test_precision += results['precision']
                test_recall += results['recall']
                test_f1 += results['f1']
        print(f'Epoch {epoch+1}/{args.epochs}, Test Loss: {test_loss/len(test_dataloader)}, Test Accuracy: {test_accuracy/len(test_dataloader)}, '
              f'Test Precision: {test_precision/len(test_dataloader)}, Test Recall: {test_recall/len(test_dataloader)}, Test F1: {test_f1/len(test_dataloader)}')
        if test_precision/len(test_dataloader) > best_test_precision:
            best_test_accuracy = test_accuracy/len(test_dataloader)
            best_test_precision = test_precision/len(test_dataloader)
            best_test_recall = test_recall/len(test_dataloader)
            best_test_f1 = test_f1/len(test_dataloader)
        print(f'Best Test Accuracy: {best_test_accuracy}, Best Test Precision: {best_test_precision}, Best Test Recall: {best_test_recall}, Best Test F1: {best_test_f1}')

        wandb.log({'train_loss': train_loss/len(train_dataloader),
                     'train_precision': train_precision/len(train_dataloader),
                     'test_loss': test_loss/len(test_dataloader),
                     'test_accuracy': test_accuracy/len(test_dataloader),
                     'test_precision': test_precision/len(test_dataloader),
                     'test_recall': test_recall/len(test_dataloader),
                     'test_f1': test_f1/len(test_dataloader),
                     'best_test_accuracy': best_test_accuracy,
                     'best_test_precision': best_test_precision,
                     'best_test_recall': best_test_recall,
                     'best_test_f1': best_test_f1})
        
    print('Training finished.')


def parse_args():
    parser = argparse.ArgumentParser(description='Association Synthesized Data')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default='Exp_Input:box+wrist', help='Experiment name')
    parser.add_argument('--data_path', type=str, default='../User Study Data/Synth')
    parser.add_argument('--window_size', type=int, default=60, help='Window size')
    parser.add_argument('--hands', type=str, default='left', help='Hands to use: left, right, both')
    parser.add_argument('--cuda', type=int, default=0, help='GPU ID')

    parser.add_argument('--imu_in_channels', type=int, default=6, help='Number of channels in IMU')
    parser.add_argument('--imu_group_norm', type=int, default=2)
    parser.add_argument('--cam_group_norm', type=int, default=3)
    parser.add_argument('--imu_out_channels', type=int, default=32, help='Number of channels in IMU')
    parser.add_argument('--kp_in_channels', type=int, default=6, help='Number of channels in keypoint')
    parser.add_argument('--box_in_channels', type=int, default=4, help='Number of channels in box')
    parser.add_argument('--cam_out_channels', type=int, default=32, help='Number of channels in CAM')
    parser.add_argument('--size_embeddings', type=int, default=128, help='Size of embeddings')
    parser.add_argument('--projection_dim', type=int, default=64, help='Projection dimension')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

    # Arguments for Contrastive Loss
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--symmetric_loss', default=True, type=bool)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.cuda = 0
    args.name = f'Exp_Input:shoulder_{args.hands}'
    args.cam_group_norm = 1
    args.kp_in_channels = 2
    main(args)
