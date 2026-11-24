import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom

from Data_Processing.preprocess import load_nifti, resample, zscore, TARGET_SPACING
from model import UNet3D
from model_rmau import ResSEMABUNet3D


def load_model(ckpt_path, device, arch=None):
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["model"]
    if arch is None:
        arch = ck.get("arch") or ("resmab" if any(k.startswith(("enc.", "final_mab")) for k in sd) else "unet")
    if arch == "resmab":
        from model_rmau import ResSEMABUNet3D
        model = ResSEMABUNet3D(1, 3)
    else:
        attn = ck.get("attention", any(k.startswith("decoder.ag") for k in sd))
        model = UNet3D(3, attention=attn)
    model.load_state_dict(sd)
    print("arsitektur:", arch, flush=True)
    return model.eval().to(device), tuple(ck["patch"])


def gaussian_map(patch, sigma_scale=1 / 8):
    """Bobot per voxel: tinggi di tengah patch, rendah di tepi (tepi prediksinya kurang andal)."""
    g = np.ones(patch, dtype=np.float32)
    for ax, p in enumerate(patch):
        c = np.arange(p) - (p - 1) / 2
        shape = [1, 1, 1]; shape[ax] = -1
        g = g * np.exp(-0.5 * (c / (p * sigma_scale)) ** 2).reshape(shape)
    return np.clip(g / g.max(), 1e-3, None)


def _starts(dim, p, step):
    if dim <= p:
        return [0]
    s = list(range(0, dim - p + 1, step))
    if s[-1] != dim - p:
        s.append(dim - p)                 # pastikan tepi akhir tertutup
    return s


@torch.no_grad()
def sliding_window(model, vol, patch, device, overlap=0.5, n_classes=3):
    shape0 = vol.shape
    pad = [max(0, p - d) for p, d in zip(patch, shape0)]
    v = np.pad(vol, [(0, q) for q in pad], mode="constant")      # pad kalau volume lebih kecil dari patch
    step = [max(1, int(p * (1 - overlap))) for p in patch]
    w = gaussian_map(patch)
    acc = np.zeros((n_classes,) + v.shape, np.float32)
    wsum = np.zeros(v.shape, np.float32)
    use_amp = device.type == "cuda"

    for z in _starts(v.shape[0], patch[0], step[0]):
        for y in _starts(v.shape[1], patch[1], step[1]):
            for x in _starts(v.shape[2], patch[2], step[2]):
                sl = (slice(z, z + patch[0]), slice(y, y + patch[1]), slice(x, x + patch[2]))
                xin = torch.from_numpy(np.ascontiguousarray(v[sl]))[None, None].to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(xin)
                prob = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
                acc[(slice(None),) + sl] += prob * w
                wsum[sl] += w
    prob = acc / wsum
    return prob[:, :shape0[0], :shape0[1], :shape0[2]]


def _fit(arr, shape):
    """Potong atau pad (tepi) supaya shape persis, jaga-jaga selisih pembulatan zoom."""
    out = arr[tuple(slice(0, min(a, s)) for a, s in zip(arr.shape, shape))]
    pad = [(0, s - o) for o, s in zip(out.shape, shape)]
    return np.pad(out, pad, mode="edge") if any(q for _, q in pad) else out


def to_original(prob, cropped_shape):
    """Probabilitas (ruang resample) -> label di ukuran crop asli. Resample probabilitas (linear), lalu argmax."""
    factors = np.array(cropped_shape) / np.array(prob.shape[1:])
    chans = [_fit(zoom(prob[k], factors, order=1, mode="nearest"), cropped_shape)
             for k in range(prob.shape[0])]
    return np.argmax(np.stack(chans), axis=0).astype(np.uint8)


def predict_nifti(in_path, out_path, ckpt_path, device=None, arch=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, patch = load_model(ckpt_path, device, arch)

    ref = sitk.ReadImage(str(in_path))                  # untuk menyalin geometri ke keluaran
    img, spacing = load_nifti(in_path)
    img = img.astype(np.float32)
    orig_shape = img.shape

    coords = np.argwhere(img != 0)                      # crop non-zero, sama seperti saat training
    bbox = tuple(slice(int(a), int(b)) for a, b in zip(coords.min(0), coords.max(0) + 1))
    cropped = img[bbox]

    x = zscore(resample(cropped, spacing, TARGET_SPACING, is_seg=False))
    prob = sliding_window(model, x, patch, device)
    seg_c = to_original(prob, cropped.shape)

    full = np.zeros(orig_shape, np.uint8)
    full[bbox] = seg_c                                  # tempel kembali ke ukuran asli

    out = sitk.GetImageFromArray(full)
    out.CopyInformation(ref)                            # spacing, origin, direction sama dengan masukan
    sitk.WriteImage(out, str(out_path))
    return full


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default=None, choices=["unet", "resmab"],
                    help="kosongkan = deteksi otomatis dari checkpoint")
    a = ap.parse_args()
    predict_nifti(a.inp, a.out, a.ckpt, arch=a.arch)
    print("selesai:", a.out)