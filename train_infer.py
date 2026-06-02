"""
PromptIR - Image Restoration (Rain & Snow)
Usage:
  Train:   python train_infer.py --mode train --data_root ./hw4_realse_dataset
  Infer:   python train_infer.py --mode infer --data_root ./hw4_realse_dataset --ckpt ./checkpoints/best.pth
  Both:    python train_infer.py --mode all   --data_root ./hw4_realse_dataset
"""

import argparse
import os
import math
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF
import random

# ─────────────────────────────────────────────
#  PromptIR Model
# ─────────────────────────────────────────────

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in  = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw_conv     = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dw_conv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class Attention(nn.Module):
    """Channel-wise self-attention (O(C^2), resolution-independent) — original PromptIR design."""
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads   = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv    = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.proj   = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        # Reshape: each head operates on (C/heads) channels across all spatial positions
        # Attention is over channels → complexity O(C^2), not O((HW)^2)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # attn: (B, heads, C/heads, C/heads) — channel × channel
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).reshape(B, C, H, W)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn   = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PromptBlock(nn.Module):
    def __init__(self, dim, prompt_dim=64, prompt_len=5):
        super().__init__()
        self.prompt_len   = prompt_len
        self.prompt_param = nn.Parameter(torch.randn(1, prompt_len, prompt_dim, 8, 8))
        self.linear_layer = nn.Linear(dim, prompt_len)
        self.conv         = nn.Conv2d(prompt_dim, dim, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        emb      = x.mean(dim=[-2, -1])            # (B, C)
        prompt_w = self.linear_layer(emb).softmax(dim=-1)  # (B, prompt_len)
        prompt   = self.prompt_param.expand(B, -1, -1, -1, -1)
        prompt_w = prompt_w.view(B, self.prompt_len, 1, 1, 1)
        prompt   = (prompt * prompt_w).sum(dim=1)
        prompt   = F.interpolate(prompt, size=(H, W), mode='bilinear', align_corners=False)
        return x + self.conv(prompt)


class DownSample(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * 2, 2, stride=2)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c // 2, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))


class PromptIR(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4,6,6,8], num_heads=[1,2,4,8],
                 ffn_expansion_factor=2.66, bias=False,
                 prompt_dim=64, prompt_len=5):
        super().__init__()
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, padding=1, bias=bias)

        # Encoder
        self.enc1    = nn.Sequential(*[TransformerBlock(dim,     num_heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])])
        self.prompt1 = PromptBlock(dim, prompt_dim, prompt_len)
        self.down1   = DownSample(dim)

        self.enc2    = nn.Sequential(*[TransformerBlock(dim*2,   num_heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])])
        self.prompt2 = PromptBlock(dim*2, prompt_dim, prompt_len)
        self.down2   = DownSample(dim*2)

        self.enc3    = nn.Sequential(*[TransformerBlock(dim*4,   num_heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])])
        self.prompt3 = PromptBlock(dim*4, prompt_dim, prompt_len)
        self.down3   = DownSample(dim*4)

        # Bottleneck
        self.bottleneck = nn.Sequential(*[TransformerBlock(dim*8, num_heads[3], ffn_expansion_factor, bias) for _ in range(num_blocks[3])])

        # Decoder
        self.up3    = UpSample(dim*8)
        self.reduce3 = nn.Conv2d(dim*8, dim*4, 1, bias=bias)
        self.dec3   = nn.Sequential(*[TransformerBlock(dim*4,   num_heads[2], ffn_expansion_factor, bias) for _ in range(num_blocks[2])])

        self.up2    = UpSample(dim*4)
        self.reduce2 = nn.Conv2d(dim*4, dim*2, 1, bias=bias)
        self.dec2   = nn.Sequential(*[TransformerBlock(dim*2,   num_heads[1], ffn_expansion_factor, bias) for _ in range(num_blocks[1])])

        self.up1    = UpSample(dim*2)
        self.reduce1 = nn.Conv2d(dim*2, dim,   1, bias=bias)
        self.dec1   = nn.Sequential(*[TransformerBlock(dim,     num_heads[0], ffn_expansion_factor, bias) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channels, 3, padding=1, bias=bias)

    def forward(self, x):
        x0 = self.patch_embed(x)

        e1 = self.prompt1(self.enc1(x0));  d1 = self.down1(e1)
        e2 = self.prompt2(self.enc2(d1));  d2 = self.down2(e2)
        e3 = self.prompt3(self.enc3(d2));  d3 = self.down3(e3)

        b  = self.bottleneck(d3)

        u3 = self.dec3(self.reduce3(torch.cat([self.up3(b),  e3], dim=1)))
        u2 = self.dec2(self.reduce2(torch.cat([self.up2(u3), e2], dim=1)))
        u1 = self.dec1(self.reduce1(torch.cat([self.up1(u2), e1], dim=1)))

        return self.output(u1) + x


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class RestorationDataset(Dataset):
    def __init__(self, data_root, patch_size=128, augment=True):
        self.patch_size = patch_size
        self.augment    = augment
        self.pairs      = []

        root = Path(data_root) / "train"
        for deg_img in sorted((root / "degraded").glob("*.*")):
            # degraded: rain-1000.png  →  clean: rain_clean-1000.png
            stem   = deg_img.stem    # e.g. "rain-1000"
            suffix = deg_img.suffix
            parts  = stem.split("-", 1)
            if len(parts) != 2:
                continue
            deg_type, img_id = parts
            clean_img = root / "clean" / f"{deg_type}_clean-{img_id}{suffix}"
            if clean_img.exists():
                self.pairs.append((deg_img, clean_img, deg_type))  # deg_type: 'rain' or 'snow' 

        assert len(self.pairs) > 0, f"No paired images found under {root}"
        print(f"[Dataset] Found {len(self.pairs)} training pairs.")

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, deg, cln):
        w, h = deg.size
        ps   = self.patch_size
        i    = random.randint(0, h - ps)
        j    = random.randint(0, w - ps)
        return TF.crop(deg, i, j, ps, ps), TF.crop(cln, i, j, ps, ps)

    def _augment(self, deg, cln):
        if random.random() > 0.5:
            deg, cln = TF.hflip(deg), TF.hflip(cln)
        if random.random() > 0.5:
            deg, cln = TF.vflip(deg), TF.vflip(cln)
        k = random.choice([0, 1, 2, 3])
        if k:
            deg = TF.rotate(deg, 90 * k)
            cln = TF.rotate(cln, 90 * k)
        return deg, cln

    def __getitem__(self, idx):
        deg_path, cln_path, deg_type = self.pairs[idx]
        deg = Image.open(deg_path).convert("RGB")
        cln = Image.open(cln_path).convert("RGB")
        deg, cln = self._random_crop(deg, cln)
        if self.augment:
            deg, cln = self._augment(deg, cln)
        is_rain = 1.0 if deg_type == 'rain' else 0.0
        return TF.to_tensor(deg), TF.to_tensor(cln), torch.tensor(is_rain)


class TestDataset(Dataset):
    def __init__(self, data_root):
        root = Path(data_root) / "test" / "degraded"
        self.images = sorted(root.glob("*.*"))
        assert len(self.images) > 0, f"No test images found in {root}"
        print(f"[Dataset] Found {len(self.images)} test images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path   = self.images[idx]
        img    = Image.open(path).convert("RGB")
        w, h   = img.size
        new_h  = math.ceil(h / 8) * 8
        new_w  = math.ceil(w / 8) * 8
        padded = TF.pad(img, [0, 0, new_w - w, new_h - h])
        return TF.to_tensor(padded), path.name, h, w   # key = filename with ext


# ─────────────────────────────────────────────
#  Loss & Metric
# ─────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


def calc_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val) - 10 * math.log10(mse.item())


# ─────────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────────

def save_curves(train_losses, val_psnrs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(epochs, train_losses, 'b-o', markersize=3, label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Charbonnier Loss')
    axes[0].set_title('Training Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    # PSNR curve
    axes[1].plot(epochs, val_psnrs, 'r-o', markersize=3, label='Val PSNR')
    if val_psnrs:
        best_epoch = val_psnrs.index(max(val_psnrs)) + 1
        axes[1].axvline(x=best_epoch, color='g', linestyle='--',
                        label=f'Best @ ep{best_epoch} ({max(val_psnrs):.2f}dB)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Validation PSNR Curve')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved training curves → {path}")


def save_psnr_histogram(val_psnrs, out_dir):
    """Histogram of per-epoch PSNR — acts like a 'distribution' view."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(val_psnrs, bins=20, color='steelblue', edgecolor='black')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Frequency (epochs)')
    plt.title('PSNR Score Distribution Across Epochs')
    plt.tight_layout()
    path = os.path.join(out_dir, 'psnr_histogram.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Plot] Saved PSNR histogram → {path}")


# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Dataset
    full_ds  = RestorationDataset(args.data_root, patch_size=args.patch_size)
    val_len  = max(1, int(len(full_ds) * 0.1))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    # Model
    model = PromptIR(dim=args.dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Parameters: {n_params:,}")

    criterion = CharbonnierLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()
    accum_steps = args.accum_steps
    rain_weight = args.rain_weight

    best_psnr    = 0.0
    train_losses = []
    val_psnrs    = []

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        for step, (deg, cln, is_rain) in enumerate(train_loader, 1):
            deg, cln = deg.to(device), cln.to(device)
            is_rain   = is_rain.to(device)  # (B,)

            with autocast():
                pred = model(deg)
                loss_per_sample = criterion(pred, cln)
                # Apply rain weight: rain samples get higher loss weight
                weight = 1.0 + (rain_weight - 1.0) * is_rain.mean()
                loss   = loss_per_sample * weight / accum_steps

            scaler.scale(loss).backward()

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps  # unscale for logging
            if step % args.log_every == 0:
                print(f"  Ep[{epoch}/{args.epochs}] Step[{step}/{len(train_loader)}] "
                      f"Loss: {total_loss/step:.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ── Validation (every epoch) ──
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for deg, cln, _ in val_loader:
                deg, cln = deg.to(device), cln.to(device)
                with autocast():
                    pred = model(deg)
                pred = pred.float().clamp(0, 1)
                val_psnr += calc_psnr(pred, cln)
        val_psnr /= len(val_loader)
        val_psnrs.append(val_psnr)

        print(f"Epoch [{epoch}/{args.epochs}]  Loss: {avg_loss:.4f}  |  Val PSNR: {val_psnr:.2f} dB"
              + (" ← best" if val_psnr > best_psnr else ""))

        # ── Save best ──
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "psnr": best_psnr},
                       os.path.join(args.ckpt_dir, "best.pth"))

        # ── Save latest ──
        torch.save({"epoch": epoch, "state_dict": model.state_dict()},
                   os.path.join(args.ckpt_dir, "latest.pth"))

        # ── Update plots every epoch ──
        save_curves(train_losses, val_psnrs, args.ckpt_dir)

    # Final histogram
    save_psnr_histogram(val_psnrs, args.ckpt_dir)

    print(f"\n[Train] Done!  Best Val PSNR: {best_psnr:.2f} dB")
    return os.path.join(args.ckpt_dir, "best.pth")


# ─────────────────────────────────────────────
#  Inference
# ─────────────────────────────────────────────

def infer(args, ckpt_path=None):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Using device: {device}")

    ckpt_path = ckpt_path or args.ckpt
    assert ckpt_path and os.path.exists(ckpt_path), \
        f"Checkpoint not found: {ckpt_path}"

    model = PromptIR(dim=args.dim).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Infer] Loaded checkpoint: {ckpt_path}")

    test_ds     = TestDataset(args.data_root)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    results = {}
    with torch.no_grad():
        for tensor, fname, orig_h, orig_w in test_loader:
            tensor = tensor.to(device)
            with autocast():
                pred = model(tensor)
            pred = pred.float().clamp(0, 1)

            # Crop back to original size
            h, w = orig_h.item(), orig_w.item()
            pred = pred[:, :, :h, :w]

            # (3, H, W) uint8  ← matches example_img2npz.py format
            img_np = (pred[0].cpu().numpy() * 255).astype(np.uint8)
            key    = fname[0]          # e.g. "rain-1.png"
            results[key] = img_np
            print(f"  {key}  shape={img_np.shape}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pred.npz")
    np.savez(out_path, **results)
    print(f"\n[Infer] Saved {len(results)} images → {out_path}")
    print(f"        Keys like: {list(results.keys())[:3]}")
    print(f"        Shape: {next(iter(results.values())).shape}, dtype: {next(iter(results.values())).dtype}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PromptIR Image Restoration")
    p.add_argument("--mode",        type=str,   default="train",
                   choices=["train","infer","all"])
    p.add_argument("--data_root",   type=str,   default="./hw4_realse_dataset")
    p.add_argument("--ckpt_dir",    type=str,   default="./checkpoints_v3")
    p.add_argument("--ckpt",        type=str,   default=None)
    p.add_argument("--output_dir",  type=str,   default="./output_v3")
    p.add_argument("--dim",         type=int,   default=48)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--patch_size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--log_every",   type=int,   default=50)
    p.add_argument("--gpu",         type=int,   default=0,
                   help="GPU index to use (e.g. 0 or 1)")
    p.add_argument("--accum_steps",  type=int,   default=4,
                   help="Gradient accumulation steps (simulates larger batch)")
    p.add_argument("--rain_weight",  type=float, default=1.5,
                   help="Loss weight multiplier for rain images (default 1.5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    elif args.mode == "all":
        best_ckpt = train(args)
        infer(args, ckpt_path=best_ckpt)