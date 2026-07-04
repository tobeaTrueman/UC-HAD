import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from tqdm import tqdm
from utils.utils import (load_hsi, normalize_hsi, extract_patches,
                          aggregate_patches_to_map, compute_pixel_auc,
                          save_rgb_image, visualize_and_save, set_seed, visualize_recon_hsi, normalize_map)
from models.teacher import OrginalUNet, DeeperUNet
import matplotlib.pyplot as plt
from dataset.PatchDataset import PatchDataset
from torch.utils.data import Dataset, DataLoader
from torch import nn
from scipy import ndimage
from siss import ssai_mask_2

#处理输入数据
def hsi_loader(args):
    data, gt = load_hsi(args.input)
    assert data is not None, "HSI data not found"
    #data = data.transpose(1, 2, 0)
    data = normalize_hsi(data, per_band = False)
    H, W, B = data.shape
    #B, H, W = data.shape
    print(f"load HSI {H} x {W} x {B}")

    #提取patch
    patches, coords = extract_patches(data, patch_size=args.patch, stride=args.stride)
    print("patches:", patches.shape, "coords:", len(coords))

    return patches, coords, data, gt, H, W, B

#计算不确定性
def compute_uncertainty(model, x, num_samples=10):
    """MC Dropout: 采样计算 mean_recon 和 uncertainty"""
    model.train()  # 启用 Dropout（即使推理）
    recons = []
    for _ in range(num_samples):
        with torch.no_grad():  # 无梯度
            recon = model(x)  # 单次重建
            recons.append(recon)
    recons = torch.stack(recons)  # (N, B, C, H, W)，如 (10, 1, B, 32, 32)
    mean_recon = recons.mean(dim=0)  # (B, C, H, W)，平均重建
    uncertainty = recons.var(dim=0).mean(dim=1)  # (B, H, W)，通道平均方差
    #print(uncertainty.shape)
    return mean_recon, uncertainty

#A模型训练
def train_A(args, iter,recon=None):
    #数据
    patches, coords, _,gt, H, W, B= hsi_loader(args)
    #_, H, W, B = patches.shape
    #print(B)
    if recon is None:
        dataset = PatchDataset(patches)
    else:
        dataset = PatchDataset(recon)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    #设备
    device = torch.device('cuda' if torch.cuda.is_available()and not args.no_cuda else 'cpu')
    #模型
    model = DeeperUNet(in_ch=B).to(device)
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #损失函数
    criterion = nn.MSELoss()
    #开始训练
    best_loss = 1e9

    #if os.path.exists(os.path.join(args.ckpt_A, 'best_model.pth')):

    #print("best_model.pth not found, starting training...")
    for epoch in range(1, args.epochs_A+1):
        model.train()
        running_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            batch = batch.to(device)
            recon =model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs_A} loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.ckpt_A, f'best_model_{iter}.pth'))

    print("Training finished, best loss:", best_loss)

    #print(f"Getting the Uncertainty Map!!! Loading checkpoint from {os.path.join(args.ckpt, 'best_model.pth')}")
    #else:
        
    model_ckpt = torch.load(os.path.join(args.ckpt_A ,f'best_model_{iter}.pth'), map_location=device)
    #infer_model = DeeperUNet(in_ch=B).to(device)

    if 'model' in model_ckpt:
        model.load_state_dict(model_ckpt['model'])
    else:
        model.load_state_dict(model_ckpt)

    patch_uncertainties = []
    num_samples = 10

    with torch.no_grad():
        for i in tqdm(range(patches.shape[0])):
            p = patches[i]
            x = np.transpose(p, (2, 0, 1))[None,...] #1, B, H, W
            x = torch.from_numpy(x.astype('float32')).to(device)

            _, uncertainty = compute_uncertainty(model, x, num_samples)
            uncertainty = uncertainty.cpu().numpy()[0]
            patch_uncertainties.append(uncertainty.astype('float32'))
    
    patch_uncertainties = np.stack(patch_uncertainties, axis=0)  
    unc_map = aggregate_patches_to_map(patch_uncertainties, coords, H, W, patch_size=args.patch)
  
    #save 
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, f'unc_map_{iter}.npy'), unc_map)
    unc_map_norm = (unc_map - unc_map.min()) / (unc_map.max() - unc_map.min() + 1e-9)
    plt.imsave(os.path.join(args.out, f'unc_map_{iter}.png'), unc_map_norm, cmap='jet')
    print("Saved results to", args.out)

#获取B重建的的原图, 接近背景，没毛病
def get_reconhsi(args, iter):

    patches, coords, _, gt, H, W, B= hsi_loader(args)

    device = torch.device('cuda' if torch.cuda.is_available()and not args.no_cuda else 'cpu')
    model_ckpt = torch.load(os.path.join(args.ckpt_B, f'best_model_{iter}.pth'), map_location=device)

    infer_model = OrginalUNet(in_ch = B).to(device)

    if 'model' in model_ckpt:
        infer_model.load_state_dict(model_ckpt['model'])
    else:
        infer_model.load_state_dict(model_ckpt)
    infer_model.eval()

    recon_patches = []
    with torch.no_grad():
        for i in tqdm(range(patches.shape[0])):
            p = patches[i]
            x = np.transpose(p, (2, 0, 1))[None,...] #1, B, H, W
            x = torch.from_numpy(x.astype('float32')).to(device)

            recon = infer_model(x)
            recon = recon.cpu().numpy()[0] #H, W, B
            recon = np.transpose(recon, (1, 2, 0))#H, W, B
            recon_patches.append(recon.astype('float32'))

        recon_patches = np.stack(recon_patches, axis=0)
    
    recon_hsi = np.zeros((H, W, B), dtype='float32')
    for b in range(B):
        band_patches = recon_patches[:, :, :, b]
        recon_hsi[:, :, b] = aggregate_patches_to_map(band_patches, coords, H, W, patch_size=args.patch)
    
    #visualize_recon_hsi(recon_hsi, './recon_hsi.png', rgb_bands=(29,19,9))
    
    return recon_patches, recon_hsi

def getmaskedhsi(args, iter, recon_hsi=None):
    
    if recon_hsi is None:
        _, _, data, gt, H, W, B = hsi_loader(args)
    else:
        _, _, _, gt, H, W, B = hsi_loader(args)
        data = recon_hsi
        patches, coords = extract_patches(recon_hsi, patch_size=args.patch, stride=args.stride)
        print("Recon_patches:", patches.shape, "coords:", len(coords))
    um= np.load(os.path.join(args.out, f'unc_map_{iter}.npy'))
    um = normalize_map(um)
    # threshold = np.percentile(um, 99.9)
    # mask = (um > threshold).astype(np.uint8)

    # # 使用二值膨胀把每个 mask 点扩展为 win_size x win_size 窗口
    # footprint = np.ones((3, 3), dtype=np.uint8)
    # expanded = ndimage.convolve(mask.astype(float), footprint, mode='constant')
    # expanded = (expanded > 0).astype(np.uint8)

    # masked_hsi = data.copy()
    # mean_val = data.mean(axis=(0,1))
    # masked_hsi[expanded == 1] = mean_val

    masked_hsi, _ = ssai_mask_2(data, um)

    patches, coords = extract_patches(masked_hsi, patch_size=args.patch, stride=args.stride)
    print("patches:", patches.shape, "coords:", len(coords))

    return patches, coords, gt, H, W, B

#B模型训练
def train_B(args, iter, recon_hsi=None):
    #patches, coords, gt, H, W, B= hsi_loader(args)
    patches, coords, gt, H, W, B = getmaskedhsi(args, iter, recon_hsi)

    dataset = PatchDataset(patches)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available()and not args.no_cuda else 'cpu')
    model = OrginalUNet(in_ch = B).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = 1e9

    for epoch in range(1, args.epochs_B+1):
        model.train()
        running_loss = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
                batch = batch.to(device)
                recon =model(batch)
                loss = criterion(recon, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs_B} loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(args.ckpt_B, f'best_model_{iter}.pth'))

    print("Training finished, best loss:", best_loss)

#循环
def cycle(args):
    set_seed(args.seed)

    recon = None
    recon_hsi = None
    for iter in range(args.iters):
        train_A(args, iter,recon)    
        train_B(args, iter, recon_hsi)
        recon, recon_hsi = get_reconhsi(args, iter)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/HYDICE.mat', help='path to .mat or .npy HSI file')
    parser.add_argument('--ckpt_A', type=str, default='./ckpts/cycle/HYDICE/model_A', help='output dir')
    parser.add_argument('--ckpt_B', type=str, default='./ckpts/cycle/HYDICE/model_B', help='output dir')
    parser.add_argument('--out', type=str, default='./img/cycle/HYDICE', help='output dir')
    parser.add_argument('--patch', type=int, default=32, help='patch size')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs_A', type=int, default=150)
    parser.add_argument('--epochs_B', type = int, default=200)
    parser.add_argument('--iters', type=int, default=8)
    #parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_patches', type=int, default=2000, help='subsample patches to speed training (0 means use all)')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    # 检查并创建必要的目录
    for dir_path in [args.ckpt_A, args.ckpt_B, args.out]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

    cycle(args)



