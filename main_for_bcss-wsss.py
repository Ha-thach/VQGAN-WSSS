import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from taming.models.vqgan_nopl import VQModel
from taming.data.bcss_wsss import BCSSWSSS_TrainDataset

import torch
import os

def save_checkpoint(path, model, opt_ae, opt_disc, epoch, global_step,
                    ddconfig=None, lossconfig=None, scaler=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "opt_ae": opt_ae.state_dict(),
        "opt_disc": opt_disc.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "ddconfig": ddconfig,
        "lossconfig": lossconfig,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)
    print(f"✅ Saved checkpoint to {path}")

def load_checkpoint(path, model, opt_ae=None, opt_disc=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if opt_ae is not None and "opt_ae" in ckpt:
        opt_ae.load_state_dict(ckpt["opt_ae"])
    if opt_disc is not None and "opt_disc" in ckpt:
        opt_disc.load_state_dict(ckpt["opt_disc"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    print(f"🔁 Loaded checkpoint from {path} (epoch {epoch}, step {global_step})")
    return epoch, global_step

def train(args):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    device = torch.device("mps" if use_mps else "cpu")
    print("Using device:", device)

    # Transform: chuẩn hoá [-1,1]
    transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    dataset = BCSSWSSS_TrainDataset(img_root=args.img_root, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    ddconfig = {
        "double_z": False,
        "z_channels": 256,
        "resolution": args.image_size,    # dùng args
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": (1, 2, 4, 8),
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "dropout": 0.0,
        "resamp_with_conv": True
    }

    lossconfig = {
        "disc_start": 2000, # sau bấy nhiêu bước mới bắt đầu dùng loss của discriminator, để tránh việc discriminator quá mạnh lúc đầu, phụ thuộc vào độ lớn của dataset
        "codebook_weight": 1.0, 
        "pixelloss_weight": 1.0,
        "disc_num_layers": 3,
        "disc_in_channels": 3,
        "disc_factor": 1.0,
        "disc_weight": 0.75,
        "perceptual_weight": 1.0,
        "use_actnorm": False,
        "disc_conditional": False,
        "disc_ndf": 64,
        "disc_loss": "hinge"
    }

    model = VQModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=args.n_embed,
        embed_dim=args.embed_dim
    ).to(device)

    last_layer = model.get_last_layer()  # dùng cho adaptive weight
    opt_ae = torch.optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()) +
        list(model.quantize.parameters()) +
        list(model.quant_conv.parameters()) +
        list(model.post_quant_conv.parameters()),
        lr=args.lr, betas=(0.5, 0.9)
    )
    opt_disc = torch.optim.Adam(model.loss.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.9))

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, global_step = load_checkpoint(args.resume, model, opt_ae, opt_disc, map_location=device)

    steps_per_epoch = len(train_loader)
    LOG_EVERY = max(1, args.log_every)

    try:
        model.train()
        for epoch in range(start_epoch, args.epochs):
            t_epoch0 = time.time()
            for step, batch in enumerate(train_loader):
                t0 = time.time()

                x = batch["image"].to(device)

                # ===== GEN / AE step =====
                xrec, qloss = model(x)
                aeloss, log_ae = model.loss(
                    qloss, x, xrec, optimizer_idx=0, global_step=global_step,
                    last_layer=last_layer, split="train"
                )
                opt_ae.zero_grad(set_to_none=True)
                aeloss.backward()
                opt_ae.step()

                # ===== DISC step =====
                discloss, log_disc = model.loss(
                    qloss.detach(), x, xrec.detach(),
                    optimizer_idx=1, global_step=global_step,
                    last_layer=last_layer, split="train"
                )
                opt_disc.zero_grad(set_to_none=True)
                discloss.backward()
                opt_disc.step()

                # ---- log ----
                if (step + 1) % LOG_EVERY == 0:
                    rec = log_ae.get('train/rec_loss', torch.tensor(0., device=device)).item()
                    ql  = log_ae.get('train/quant_loss', torch.tensor(0., device=device)).item()
                    print(f"[ep {epoch} {step+1}/{steps_per_epoch}] "
                          f"AE_loss:{aeloss.item():.4f}  D_loss:{discloss.item():.4f}  "
                          f"Recon:{rec:.4f}  Quant:{ql:.4f}  "
                          f"time:{time.time()-t0:.2f}s")

                # ---- periodic checkpoint ----
                if args.save_every > 0 and (global_step + 1) % args.save_every == 0:
                    path = os.path.join(args.save_dir, f"step_{global_step+1}.pth")
                    save_checkpoint(path, model, opt_ae, opt_disc, epoch, global_step+1, ddconfig, lossconfig)

                global_step += 1

            # ---- end-epoch checkpoint ----
            last_path = os.path.join(args.save_dir, f"last_epoch_{epoch}.pth")
            save_checkpoint(last_path, model, opt_ae, opt_disc, epoch, global_step, ddconfig, lossconfig)
            print(f"⏱ epoch {epoch} took {(time.time()-t_epoch0)/60:.2f} min")

    except KeyboardInterrupt:
        # save interrupt checkpoint
        path = os.path.join(args.save_dir, f"interrupt_step_{global_step}.pth")
        save_checkpoint(path, model, opt_ae, opt_disc, epoch, global_step, ddconfig, lossconfig)
        print(" Training interrupted by user. Checkpoint saved.")
        raise


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--img_root", type=str, default="data/sub_BCSS_WSSS/training")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_embed", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--perc_w", type=float, default=1.0, help="perceptual_weight (LPIPS)")
    parser.add_argument("--disc_start", type=int, default=2000, help="warmup steps before enabling GAN")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=1000, help="save checkpoint every N steps (0=disable)")
    parser.add_argument("--resume", type=str, default="", help="path to .pth to resume")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)