import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import torch
from tqdm import tqdm
from utils.utils import (load_hsi, normalize_hsi, extract_patches,
                          aggregate_patches_to_map, compute_pixel_auc,
                          save_rgb_image, visualize_and_save, set_seed,
                          visualize_recon_hsi)
from models.student import StudentAE, StudentUNet
from models.teacher import OrginalUNet
import time

def aggregate_patches_to_hsi(recon_patches, coords, H, W, patch_size=32):
    """将重叠 patch 重建结果平均融合回整幅高光谱图像。"""
    B = recon_patches.shape[-1]
    recon_sum = np.zeros((H, W, B), dtype='float32')
    weight = np.zeros((H, W, 1), dtype='float32')

    for idx, (i, j) in enumerate(coords):
        recon_sum[i:i+patch_size, j:j+patch_size, :] += recon_patches[idx]
        weight[i:i+patch_size, j:j+patch_size, :] += 1.0

    weight[weight == 0] = 1.0
    return recon_sum / weight

def infer(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data, gt = load_hsi(args.input)
    print(data.shape, gt.shape)
    assert data is not None
    #data = data.transpose(1, 2, 0)
    data = normalize_hsi(data, per_band=False)
    H,W,B = data.shape
    patches, coords = extract_patches(data, patch_size=args.patch, stride=args.stride)
    print("patches:", patches.shape)

    #load model 要定义模型对象
    model_ckpt = torch.load(args.model, map_location=device)
    #model = Teachermodel(in_ch=B).to(device)
    model = OrginalUNet(in_ch=B).to(device)

    if 'model' in model_ckpt:
        model.load_state_dict(model_ckpt['model'])
    else:
        model.load_state_dict(model_ckpt)
    model.eval()

    #for each patch compute per-pixel reconstruction error
    patch_errors = []
    recon_patches = []
    start = time.perf_counter()
    with torch.no_grad():
        for i in tqdm(range(patches.shape[0])):
            p = patches[i]
            x = np.transpose(p, (2, 0, 1))[None,...] #1, B, H, W
            x = torch.from_numpy(x.astype('float32')).to(device)

            recon = model(x)
            recon = recon.cpu().numpy()[0] #H, W, B
            recon = np.transpose(recon, (1, 2, 0))#H, W, B
            recon_patches.append(recon.astype('float32'))

            #计算每个像素的L2损失 across bands
            err = np.linalg.norm(recon-p, axis=2) #H, W
            patch_errors.append(err.astype('float32'))
    
    patch_errors = np.stack(patch_errors, axis=0) #(N, p, p)
    recon_patches = np.stack(recon_patches, axis=0) 
    print(recon_patches.shape)
    score_map = aggregate_patches_to_map(patch_errors, coords, H, W, patch_size=args.patch)
    end = time.perf_counter()
    print(f"运行耗时：{end - start:.6f} 秒")
    recon_hsi = aggregate_patches_to_hsi(recon_patches, coords, H, W, patch_size=args.patch)
    

    # normalize rgb for saving
    rgb = data[:, : ,(29, 19, 9)]
    rgb = (rgb-rgb.min()) / (rgb.max() - rgb.min() + 1e-9)

    #compute AUC
    auc = compute_pixel_auc(score_map, gt)
    print("Pixel AUC:", auc)

    #save 
    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, 'score_map.npy'), score_map)
    np.save(os.path.join(args.out, 'recon_hsi.npy'), recon_hsi)
    visualize_and_save(rgb, score_map, gt, args.out, prefix='san_diego')
    recon_rgb_bands = tuple(min(idx, B - 1) for idx in (29, 19, 9))
    visualize_recon_hsi(
        recon_hsi,
        os.path.join(args.out, 'san_diego_recon_rgb.png'),
        rgb_bands=recon_rgb_bands
    )
    with open(os.path.join(args.out, 'metrics.txt'), 'w') as f:
        f.write(f"AUC: {auc}\n")
    print("Saved results to", args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/lijicai/distillation/data/SanDiego.mat',required=False)
    parser.add_argument('--model', type=str, default='/home/lijicai/distillation/ckpts/cycle/SanDiego/model_B/best_model_5.pth',required=False, help='path to model .pth (best_model.pth)')
    parser.add_argument('--out', type=str, default='/home/lijicai/distillation/img/test_imgs')
    parser.add_argument('--patch', type=int, default=32)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    #parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
   
    infer(args)
   

    






