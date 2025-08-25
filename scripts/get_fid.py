import os, json, numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob

# FID
from torch_fidelity import calculate_metrics as _tf_calculate

# ---- DeepFace 懒加载：禁用 TensorFlow GPU，只在用到时导入 ----
def _get_deepface_cpu():
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')  # 必须在任何 TF GPU 初始化前执行
    except Exception:
        pass
    from deepface import DeepFace as _DF
    return _DF

def _deepface_embed(np_img_uint8, model_name):
    DF = _get_deepface_cpu()
    reps = DF.represent(
        img_path=np_img_uint8[..., ::-1],  # RGB->BGR
        model_name=model_name,
        detector_backend='skip',
        enforce_detection=False,
        align=False
    )
    if isinstance(reps, list) and len(reps) > 0 and 'embedding' in reps[0]:
        return np.asarray(reps[0]['embedding'], dtype=np.float32)
    raise RuntimeError(f'Embedding failed: {model_name}')

class CoupledImageDataset(Dataset):
    # 左半 GT，右半 Gen；递归搜图、大小写无关，并排序，保证可复现
    def __init__(self, data_dir):
        exts = ('*.png','*.jpg','*.jpeg','*.PNG','*.JPG','*.JPEG')
        paths = []
        for e in exts:
            paths += glob(os.path.join(data_dir, '**', e), recursive=True)
        self.image_paths = sorted(paths)
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        w, h = img.size
        gt  = np.asarray(img.crop((0, 0,    w//2, h)), dtype=np.float32) / 255.0
        gen = np.asarray(img.crop((w//2, 0, w,    h)), dtype=np.float32) / 255.0
        return gen, gt, os.path.basename(p)

# 懒读盘版本，返回 uint8 CHW（0..255）
class _GTOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        w, h = img.size
        gt = np.asarray(img.crop((0, 0, w // 2, h)), dtype=np.uint8)  # 注意：uint8，不做/255
        return torch.from_numpy(gt).permute(2, 0, 1)  # CHW, dtype=torch.uint8

class _GenOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        w, h = img.size
        gen = np.asarray(img.crop((w // 2, 0, w, h)), dtype=np.uint8)  # uint8
        return torch.from_numpy(gen).permute(2, 0, 1)  # CHW, uint8


def _cosine(a,b):
    return 1.0 - (a @ b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12)
def _l2(a,b): return float(np.linalg.norm(a-b))

def main(args):
    # Inception 权重：两种离线路径都支持
    if args.compute_fid:
        cand = [
            os.path.join(args.inception_weights_dir, 'hub', 'checkpoints', 'weights-inception-2015-12-05-6726825d.pth'),
            os.path.join(args.inception_weights_dir, 'checkpoints', 'inception-2015-12-05.pt'),
        ]
        inc = next((p for p in cand if os.path.isfile(p)), None)
        if inc is None:
            raise FileNotFoundError(
                "未找到 Inception 权重，请二选一：\n"
                f"1) {cand[0]}\n"
                f"2) {cand[1]}"
            )
        os.environ['TORCH_HOME'] = args.inception_weights_dir

    # DeepFace 权重（该版本默认查 $DEEPFACE_HOME/.deepface/weights/）
    if args.compute_vgg or args.compute_dlib:
        vgg_path  = os.path.join(args.deepface_weights_dir, '.deepface', 'weights', 'vgg_face_weights.h5')
        dlib_path = os.path.join(args.deepface_weights_dir, '.deepface', 'weights', 'dlib_face_recognition_resnet_model_v1.dat')
        if args.compute_vgg and not os.path.isfile(vgg_path):
            raise FileNotFoundError(f'缺少 {vgg_path}')
        if args.compute_dlib and not os.path.isfile(dlib_path):
            raise FileNotFoundError(f'缺少 {dlib_path}')
        os.environ['DEEPFACE_HOME'] = args.deepface_weights_dir

    ds = CoupledImageDataset(args.data_path)
    if len(ds) == 0:
        raise RuntimeError(f"目录里没找到图片：{args.data_path}（支持子目录；支持 .png/.jpg/.jpeg；大小写不敏感）")

    # DataLoader 仅供 DeepFace 遍历；FID 走懒读盘 Dataset
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    vgg_hits, dlib_hits = [], []

    for gen, gt, _ in tqdm(dl, total=len(ds)):
        gen = gen.squeeze(0).numpy(); gt = gt.squeeze(0).numpy()

        if args.compute_vgg:
            e_gt  = _deepface_embed((gt*255).astype(np.uint8),  'VGG-Face')
            e_gen = _deepface_embed((gen*255).astype(np.uint8), 'VGG-Face')
            if args.vgg_metric == 'cosine':
                dist = _cosine(e_gt, e_gen); thr = args.vgg_thr if args.vgg_thr is not None else 0.68
            else:
                dist = _l2(e_gt, e_gen);     thr = args.vgg_thr if args.vgg_thr is not None else 1.17
            vgg_hits.append(float(dist <= thr))

        if args.compute_dlib:
            e_gt  = _deepface_embed((gt*255).astype(np.uint8),  'Dlib')
            e_gen = _deepface_embed((gen*255).astype(np.uint8), 'Dlib')
            if args.dlib_metric == 'cosine':
                dist = _cosine(e_gt, e_gen); thr = args.dlib_thr if args.dlib_thr is not None else 0.23
            else:
                dist = _l2(e_gt, e_gen);     thr = args.dlib_thr if args.dlib_thr is not None else 0.07
            dlib_hits.append(float(dist <= thr))

    metrics_out = {}

    if args.compute_fid:
        gt_ds  = _GTOnlyDataset(ds.image_paths)
        gen_ds = _GenOnlyDataset(ds.image_paths)
        m = _tf_calculate(
            input1=gt_ds, input2=gen_ds,
            cuda=torch.cuda.is_available(),
            batch_size=args.fid_batch,
            dataloader_num_workers=0,   # 避免 worker OOM
            save_cpu_ram=True,          # 降低内存占用
            isc=False, fid=True, kid=False, verbose=False
        )
        metrics_out['FID'] = float(m['frechet_inception_distance'])
        metrics_out['n_images'] = len(ds)

    if args.compute_vgg:
        metrics_out['VGGFace_ACC'] = float(np.mean(vgg_hits)) if len(vgg_hits)>0 else 0.0
        metrics_out['VGGFace_metric'] = args.vgg_metric
        metrics_out['VGGFace_thr'] = args.vgg_thr if args.vgg_thr is not None else (0.68 if args.vgg_metric=='cosine' else 1.17)

    if args.compute_dlib:
        metrics_out['Dlib_ACC'] = float(np.mean(dlib_hits)) if len(dlib_hits)>0 else 0.0
        metrics_out['Dlib_metric'] = args.dlib_metric
        metrics_out['Dlib_thr'] = args.dlib_thr if args.dlib_thr is not None else (0.23 if args.dlib_metric=='cosine' else 0.07)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    print('✔ saved to', args.out_file)

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--data_path', required=True, help='拼接图目录（左=GT，右=Gen）')
    ap.add_argument('--inception_weights_dir', type=str, default='pretrained_models', help='Inception 权重根目录（等同 TORCH_HOME）')
    ap.add_argument('--deepface_weights_dir', type=str, default='pretrained_models', help='DeepFace 权重根目录（等同 DEEPFACE_HOME）')
    ap.add_argument('--compute_fid', action='store_true')
    ap.add_argument('--compute_vgg', action='store_true')
    ap.add_argument('--compute_dlib', action='store_true')
    ap.add_argument('--fid_batch', type=int, default=64)
    ap.add_argument('--vgg_metric', choices=['cosine','euclidean'], default='cosine')
    ap.add_argument('--dlib_metric', choices=['cosine','euclidean'], default='euclidean')
    ap.add_argument('--vgg_thr', type=float, default=None)
    ap.add_argument('--dlib_thr', type=float, default=None)
    ap.add_argument('--out_file', type=str, default='inference_metrics/metrics.json')
    args = ap.parse_args()
    main(args)
