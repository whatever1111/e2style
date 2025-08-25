import os, json
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


class CoupledImageDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Crop 左边 GT，右边 Gen
        gt_img = img.crop((0, 0, w // 2, h))
        gen_img = img.crop((w // 2, 0, w, h))

        gt_np = np.array(gt_img).astype(np.float32) / 255.0
        gen_np = np.array(gen_img).astype(np.float32) / 255.0

        return gen_np, gt_np, os.path.basename(img_path)


def main(args):
    dataset = CoupledImageDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    psnr_list, ssim_list = [], []

    for gen_np, gt_np, name in tqdm(dataloader):
        gen_np = gen_np.squeeze(0).numpy()
        gt_np = gt_np.squeeze(0).numpy()

        psnr = compare_psnr(gt_np, gen_np, data_range=1.0)
        ssim = compare_ssim(gt_np, gen_np, channel_axis=2, data_range=1.0)


        psnr_list.append(psnr)
        ssim_list.append(ssim)

    def write_result(name, values):
        out_dir = os.path.join(args.data_path, 'inference_metrics')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'stat_{name}.txt'), 'w') as f:
            f.write(f'{name.upper()} avg: {np.mean(values):.4f} ± {np.std(values):.4f}\n')
        with open(os.path.join(out_dir, f'scores_{name}.json'), 'w') as f:
            json.dump({i: float(v) for i, v in enumerate(values)}, f)

    write_result('psnr', psnr_list)
    write_result('ssim', ssim_list)
    print('[✔] PSNR and SSIM metrics saved.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to coupled 2048x1024 images')
    args = parser.parse_args()
    main(args)
