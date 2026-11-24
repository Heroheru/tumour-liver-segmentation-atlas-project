import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Data_Processing.dataset import LiverPatchDataset
from model import UNet3D
from model_rmau import ResSEMABUNet3D

 

N_CLASSES = 3   # 0 = background, 1 = liver, 2 = tumor


class DiceCELoss(nn.Module):
    """Cross-entropy + soft Dice. Dice dihitung per sampel dan per kelas, background tidak ikut."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        p = torch.softmax(logits, dim=1)                                      # (B, C, D, H, W)
        onehot = F.one_hot(target, N_CLASSES).permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        dims = (2, 3, 4)
        inter = (p * onehot).sum(dims)                                        # (B, C)
        denom = p.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return ce + (1.0 - dice[:, 1:].mean())                                # buang kelas 0


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, dl, loss_fn, device, use_amp):
    model.eval()
    tp = np.zeros(N_CLASSES); sp = np.zeros(N_CLASSES); st = np.zeros(N_CLASSES)
    total_loss, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        total_loss += loss_fn(logits, y).item(); n += 1
        pred = logits.argmax(1)
        for k in range(1, N_CLASSES):
            tp[k] += ((pred == k) & (y == k)).sum().item()
            sp[k] += (pred == k).sum().item()
            st[k] += (y == k).sum().item()
    dice = 2 * tp / (sp + st + 1e-8)          # dihitung global atas semua patch validasi
    return total_loss / max(n, 1), dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../preproc")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters", type=int, default=250, help="iterasi per epoch")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--patch", type=int, nargs=3, default=[48, 192, 224])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=12, help="jumlah kasus untuk validasi")
    ap.add_argument("--smoke", action="store_true", help="tes cepat: patch kecil, 2 epoch")
    ap.add_argument("--resume", action="store_true", help="lanjutkan dari checkpoint.pt di folder --out")
    ap.add_argument("--no_amp", action="store_true", help="matikan fp16 (kalau loss jadi NaN)")
    ap.add_argument("--log_every", type=int, default=25, help="cetak progres tiap N iterasi")
    ap.add_argument("--arch", default="unet", choices=["unet", "resmab"])


    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.iters, args.patch, args.workers, args.n_val = 2, 10, [32, 64, 64], 2, 4

    torch.manual_seed(0); np.random.seed(0)
    device = get_device()
    # ganti baris use_amp
    use_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = device.type == "cuda" 
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | amp: {use_amp} | patch: {args.patch} | epochs: {args.epochs}")

    # ---------- split latih / validasi (tetap, disimpan untuk evaluasi nanti) ----------
    split_path = out / "split.json"
    if args.resume and split_path.exists():
        split = json.load(open(split_path))
        train_ids, val_ids = split["train"], split["val"]
    else:
        ids = sorted(p.name.replace("_img.npy", "") for p in Path(args.data).glob("*_img.npy"))
        ids = [ids[i] for i in np.random.default_rng(0).permutation(len(ids))]
        val_ids, train_ids = ids[:args.n_val], ids[args.n_val:]
        json.dump({"train": train_ids, "val": val_ids}, open(split_path, "w"), indent=1)
    print(f"latih: {len(train_ids)} kasus | validasi: {len(val_ids)} kasus | Arsitektur {args.arch}")

    train_ds = LiverPatchDataset(args.data, patch_size=args.patch, iters_per_epoch=args.iters,
                                 batch_size=args.batch, augment=True, case_ids=train_ids)
    val_ds = LiverPatchDataset(args.data, patch_size=args.patch, iters_per_epoch=max(args.iters // 5, 5),
                               batch_size=args.batch, augment=False, case_ids=val_ids, seed=123)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=min(args.workers, 2))

    model = (ResSEMABUNet3D(1, N_CLASSES) if args.arch == "resmab" else UNet3D(N_CLASSES)).to(device)   
    loss_fn = DiceCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99,
                          nesterov=True, weight_decay=3e-5)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:                                   # PyTorch versi lama
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best = 0, -1.0
    ckpt_path = out / "checkpoint.pt"

    if args.resume:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resume dari epoch {start_epoch + 1} (dice terbaik sejauh ini: {best:.3f})")
        if ck["total_epochs"] != args.epochs:
            print(f"PERINGATAN: --epochs sekarang {args.epochs}, run awal {ck['total_epochs']}. "
                f"Jadwal LR poly berubah, jadi LR akan melompat.")
        if list(ck["patch"]) != list(args.patch):
            print(f"PERINGATAN: patch checkpoint {ck['patch']} berbeda dari {args.patch}.")

    log_f = open(out / "log.csv", "a" if args.resume else "w", newline="")
    log = csv.writer(log_f)

    if not args.resume:
        log.writerow(["epoch", "lr", "train_loss", "val_loss", "dice_liver", "dice_tumor", "detik"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9          # poly decay
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        run_loss, n = 0.0, 0
        win_loss, t_win = 0.0, time.time()        # untuk log per jendela
        n_iters = len(train_dl)                   # iterasi per epoch (= --iters)
        g_total = args.epochs * n_iters           # total iterasi seluruh training

        # ---------- 1. LATIH ----------
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt); scaler.update()

            l = loss.item()
            run_loss += l; n += 1
            win_loss += l

            if n % args.log_every == 0 or n == n_iters:
                k = n % args.log_every or args.log_every
                dt_it = (time.time() - t_win) / k
                g_iter = epoch * n_iters + n
                sisa = dt_it * (g_total - g_iter) / 60
                print(f"   epoch {epoch + 1}/{args.epochs} | iter {n:>3}/{n_iters} | "
                      f"total {g_iter}/{g_total} ({g_iter / g_total:.0%}) | "
                      f"loss {win_loss / k:.4f} | {dt_it:.2f} d/iter | sisa ~{sisa:.0f} menit", flush=True)
                win_loss, t_win = 0.0, time.time()

        # ---------- 2. VALIDASI (sekali per epoch, SETELAH loop latih) ----------
        val_loss, dice = validate(model, val_dl, loss_fn, device, use_amp)
        mean_dice = float(dice[1:].mean())
        dt = time.time() - t0
        log.writerow([epoch, f"{lr:.5f}", f"{run_loss / n:.4f}", f"{val_loss:.4f}",
                      f"{dice[1]:.4f}", f"{dice[2]:.4f}", f"{dt:.0f}"]); log_f.flush()
        print(f"epoch {epoch + 1:>3}/{args.epochs} | lr {lr:.4f} | train {run_loss / n:.4f} | "
              f"val {val_loss:.4f} | dice liver {dice[1]:.3f} tumor {dice[2]:.3f} | {dt:.0f} d", flush=True)
        if device.type == "cuda":
            print(f"      memori GPU puncak: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)

        # ---------- 3. SIMPAN ----------
        if mean_dice > best:
            best = mean_dice
            torch.save({"model": model.state_dict(), "epoch": epoch, "patch": args.patch}, out / "best.pt")
        save_checkpoint(ckpt_path, model, opt, scaler, epoch, best, args)

    print(f"selesai. dice validasi terbaik (rata-rata liver+tumor): {best:.3f}")

def save_checkpoint(path, model, opt, scaler, epoch, best, args):
    """Simpan lewat file sementara, supaya checkpoint tidak rusak kalau proses mati di tengah menulis."""
    tmp = Path(str(path) + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,                  # epoch yang SUDAH selesai (mulai dari 0)
        "best": best,                    # Dice validasi terbaik sampai sekarang
        "patch": args.patch,
        "total_epochs": args.epochs,
    }, tmp)
    os.replace(tmp, path)

if __name__ == "__main__":
    main()