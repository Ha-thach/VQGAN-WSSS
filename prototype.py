import os
import argparse
import json
import numpy as np
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from taming.models.vqgan_nopl import VQModel
from taming.data.bcss_wsss import BCSSWSSS_TrainDataset


CLASSES = ["TUM", "STR", "LYM", "NEC"]  # theo mô tả one-hot 4 chiều


import matplotlib.pyplot as plt
def extract_indices(info):
    # case tuple/list
    if isinstance(info, (tuple, list)):
        # theo class này: info = (perplexity, min_encodings, min_encoding_indices)
        for x in info[::-1]:  # duyệt từ cuối về đầu
            if torch.is_tensor(x) and x.dtype == torch.long:
                return x
        # hoặc đệ quy nếu lồng sâu
        for x in info:
            got = extract_indices(x)
            if got is not None:
                return got
    return None

@torch.no_grad()
def build_model_from_ckpt(ckpt_path, device, image_size, fallback_embed_dim=None, fallback_n_embed=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ddconfig = ckpt.get("ddconfig", None)
    lossconfig = ckpt.get("lossconfig", None)

    if ddconfig is None:
        # fallback nếu ckpt không lưu ddconfig
        ddconfig = {
            "double_z": False,
            "z_channels": 256,
            "resolution": image_size,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1, 2, 4, 8),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.0,
            "resamp_with_conv": True
        }
    if lossconfig is None:
        # fallback lossconfig tối thiểu (không dùng trong infer)
        lossconfig = {
            "disc_start": 2000,
            "codebook_weight": 1.0,
            "pixelloss_weight": 1.0,
            "disc_num_layers": 3,
            "disc_in_channels": 3,
            "disc_factor": 0.0,
            "disc_weight": 0.0,
            "perceptual_weight": 0.0,
            "use_actnorm": False,
            "disc_conditional": False,
            "disc_ndf": 64,
            "disc_loss": "hinge"
        }

    # Tạm tạo mô hình để biết shape codebook từ state_dict
    # Suy ra (n_embed, embed_dim) từ weight của quantizer
    sd = ckpt["model"]
    # các key thường gặp: 'quantize.embedding.weight' hoặc 'quantize.embedding'
    qkeys = [k for k in sd.keys() if "quantize" in k and "embedding" in k and "weight" in k]
    if not qkeys:
        raise RuntimeError("Không tìm thấy 'quantize.embedding.weight' trong checkpoint.")
    weight = sd[qkeys[0]]  # Tensor [n_embed, embed_dim]
    n_embed, embed_dim = weight.shape

    # cho phép fallback override nếu cần
    if fallback_embed_dim is not None: embed_dim = fallback_embed_dim
    if fallback_n_embed is not None:   n_embed = fallback_n_embed

    model = VQModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=n_embed,
        embed_dim=embed_dim
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, (n_embed, embed_dim)


@torch.no_grad()
def extract_codebook(model):
    # codebook/prototype: [n_embed, embed_dim]
    return model.quantize.embedding.weight.detach().cpu().numpy()

def save_coarse_map(idx, quant_tensor, out_png_path, cmap="tab20"):
    """
    idx_tensor: LongTensor [B*h*w] or [B,h,w] or [h*w] or [h,w]
    quant_tensor: Tensor [B,C,h,w] để suy ra h,w nếu cần
    """
    
    h, w = quant_tensor.shape[2], quant_tensor.shape[3]
    idx_hw = idx.reshape(h, w)

    plt.figure(figsize=(4,4))
    plt.imshow(idx_hw, cmap=cmap)
    plt.axis("off")
    plt.title("Code index map")
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, bbox_inches="tight", pad_inches=0)
    plt.close()

@torch.no_grad()
def eval_dataset(args):
    # device
    #use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    #device = torch.device("mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cpu")
    print("Using device:", device)
    ckpt_path = "/Users/thachha/Desktop/AIO2025-official/AIMA/CP- WSSS/taming-transformers/checkpoints/last_epoch_4.pth"
    # model
    model, (n_embed, embed_dim) = build_model_from_ckpt(
        ckpt_path, device, args.image_size,
        fallback_embed_dim=args.embed_dim, fallback_n_embed=args.n_embed
    )
    codebook = extract_codebook(model)
    print(f"Codebook shape: {codebook.shape}  (n_embed={n_embed}, embed_dim={embed_dim})")

    # dataset & loader (same transform as train)
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    ds = BCSSWSSS_TrainDataset(img_root=args.img_root, transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    print(f"Dataset size: {len(ds)} images")
    
    # thống kê
    per_class_hist = {c: np.zeros(n_embed, dtype=np.int64) for c in range(len(CLASSES))}
    major_code_to_class = {}   # mapping dựa trên training-set histogram
    rows = []                  # lưu CSV
    os.makedirs(args.out_dir, exist_ok=True)
    map_dir = os.path.join(args.out_dir, "coarse_maps")
    os.makedirs(map_dir, exist_ok=True)
    npy_dir = os.path.join(args.out_dir, "index_npy")
    os.makedirs(npy_dir, exist_ok=True)

    # với mỗi ảnh: lấy indices, histogram, z_q_mean
    for i, batch in enumerate(dl):
        x = batch["image"].to(device)             # [1,3,H,W] in [-1,1]
        labels = batch["cls_label"].numpy()[0]    # one-hot 4
        fname = batch["img_path_"][0]

        # encode → quantize
        # encode trả: quant (z_q), emb_loss, info
        quant, emb_loss, info = model.encode(x)
        # indices: tuỳ config 'sane_index_shape', thường là [B, h*w] hoặc [B,h,w]
        # Taming VQ thường lưu key 'min_encoding_indices'
        idx = extract_indices(info)
        if idx is None:
            # VectorQuantizer2 hay trả 'indices' trong info; fallback
            raise RuntimeError("Không tìm thấy indices trong info")

        print(f"Image {fname}: code indices shape: {idx.shape}, dtype: {idx.dtype}")
        print(idx) # in ra indices; tensor long giá trị 0..n_embed-1
        idx = idx.cpu().numpy().reshape(-1)  # [h*w] or [h,w] → [N]
        hist = np.bincount(idx, minlength=n_embed) # [n_embed], đếm số lần mỗi code xuất hiện
        print(hist)
        # z_q trung bình theo không gian (prototype-mean của ảnh)
        z_q = quant.detach().cpu().numpy()  # [1, C, h, w]
        z_q_mean = z_q.mean(axis=(2,3)).reshape(-1)   # [C]

        # class id (giả sử one-hot single-label)
        class_id = int(np.argmax(labels))

        # cộng histogram vào class-hist
        per_class_hist[class_id] += hist

        # lưu dòng CSV
        topk_ids = hist.argsort()[-args.topk:][::-1] #
        rows.append({
            "file": fname,
            "class": CLASSES[class_id],
            "top_codes": ",".join(map(str, topk_ids.tolist())),
            "top_counts": ",".join(map(str, hist[topk_ids].tolist()))
        })

        # LƯU coarse segmentation map & raw indices
        stem = os.path.splitext(os.path.basename(fname))[0]
        png_path = os.path.join(map_dir, f"{stem}_coarse.png")
        save_coarse_map(idx, quant, png_path, cmap="tab20")

        if args.dump_indices:
            np.save(os.path.join(npy_dir, f"{stem}_indices.npy"),
                    idx.detach().cpu().numpy())

        if args.dump_zq:
            np.save(os.path.join(args.out_dir, f"{stem}_zqmean.npy"), z_q_mean)

        if (i+1) % 20 == 0:
            print(f"[{i+1}/{len(ds)}] processed… maps→ {map_dir}")

    total_hist = np.stack([per_class_hist[c] for c in range(len(CLASSES))], axis=0)
    code_to_class = np.argmax(total_hist, axis=0)

    correct = 0
    for r in rows:
        top_codes = list(map(int, r["top_codes"].split(",")))
        mode_code = top_codes[0]
        pred_c = CLASSES[code_to_class[mode_code]]
        if pred_c == r["class"]:
            correct += 1
    acc = correct / len(rows)
    print(f"Naive major-code accuracy: {acc*100:.2f}%  ({correct}/{len(rows)})")

    csv_path = os.path.join(args.out_dir, "codes_per_image.csv")
    with open(csv_path, "w") as f:
        f.write("file,class,top_codes,top_counts\n")
        for r in rows:
            f.write(f"{r['file']},{r['class']},{r['top_codes']},{r['top_counts']}\n")
    print("Saved:", csv_path)

    np.save(os.path.join(args.out_dir, "per_class_hist.npy"), total_hist)
    with open(os.path.join(args.out_dir, "code_to_class.json"), "w") as f:
        json.dump({int(k): CLASSES[int(v)] for k, v in enumerate(code_to_class.tolist())}, f, indent=2)
    print("Saved per-class hist & code_to_class mapping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, default="data/sub_BCSS_WSSS/training")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--n_embed", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="eval_codes")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--dump_zq", action="store_true", help="save z_q mean per image")
    parser.add_argument("--dump_indices", action="store_true", help="save raw indices .npy per image")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eval_dataset(args)

