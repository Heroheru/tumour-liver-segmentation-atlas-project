"""
dataset.py
Data loader patch 3D dengan foreground oversampling dan augmentasi (versi sederhana nnU-Net).

• Masalah: Jika model dilatih dengan gambar apa adanya, model akan menerima jutaan contoh piksel atau bounding box latar belakang (misalnya jalanan, langit, dinding пусто) dan hanya sedikit contoh objek target (misalnya mobil, pejalan kaki, keretakan dinding).
• Akibat jika tidak diatasi: Model akan menjadi sangat malas dan cenderung memprediksi semua hal sebagai background karena akurasinya akan tetap terlihat tinggi secara statistik.
• Solusi: Dengan memperbanyak (oversampling) objek foreground, model dipaksa untuk lebih sering melihat dan mempelajari fitur-fitur dari objek yang justru ingin kita deteksi.


"""


from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import affine_transform, gaussian_filter
from torch.utils.data import DataLoader, Dataset

PATCH = (48, 192, 224)  # (z, y, x), dari nnUNetPlans.json


def _crop(arr, lb, size):
    """
    Ambil patch berukuran `size` mulai dari koordinat `lb`.
    lb boleh negatif atau melewati batas; bagian di luar volume diisi 0 (padding).
    """
    out = np.zeros(size, dtype=arr.dtype)
    src_lo = [max(l, 0) for l in lb]
    src_hi = [min(l + s, a) for l, s, a in zip(lb, size, arr.shape)]
    dst_lo = [sl - l for sl, l in zip(src_lo, lb)]
    dst_hi = [d + (h - l) for d, h, l in zip(dst_lo, src_hi, src_lo)]
    out[tuple(slice(a, b) for a, b in zip(dst_lo, dst_hi))] = \
        arr[tuple(slice(a, b) for a, b in zip(src_lo, src_hi))]
    return out


class LiverPatchDataset(Dataset):
    def __init__(self, data_dir, patch_size=PATCH, iters_per_epoch=250, batch_size=2,
                 fg_prob=0.33, augment=True, max_fg_voxels=10000, seed=None, case_ids=None):
        self.patch = tuple(patch_size)
        self.length = iters_per_epoch * batch_size   # nnU-Net: 1 epoch = 250 iterasi
        self.fg_prob = fg_prob
        self.augment = augment
        self.seed = seed

        data_dir = Path(data_dir)
        self.case_ids = sorted(p.name.replace("_img.npy", "") for p in data_dir.glob("*_img.npy"))
        assert len(self.case_ids) > 0, f"Tidak ada file *_img.npy di {data_dir.resolve()}"
        if case_ids is not None:                                  # <-- tambahan
            keep = set(case_ids)
            self.case_ids = [c for c in self.case_ids if c in keep]
        # mmap_mode="r": array dibaca dari disk seperlunya, tidak semua masuk RAM
        self.imgs = [np.load(data_dir / f"{c}_img.npy", mmap_mode="r") for c in self.case_ids]
        self.segs = [np.load(data_dir / f"{c}_seg.npy", mmap_mode="r") for c in self.case_ids]

        # Simpan koordinat voxel foreground per kasus per kelas (dikurangi supaya hemat memori)
        rng = np.random.default_rng(0)
        self.fg_locs = []
        for seg in self.segs:
            locs = {}
            for k in (1, 2):                          # 1 = liver, 2 = tumor
                coords = np.argwhere(seg == k)
                if len(coords) > max_fg_voxels:
                    coords = coords[rng.choice(len(coords), max_fg_voxels, replace=False)]
                if len(coords) > 0:
                    locs[k] = coords
            self.fg_locs.append(locs)

    def __len__(self):
        return self.length

    # ---------- pengambilan patch ----------
    def _sample_lb(self, shape, locs, rng, force_fg):
        """Tentukan sudut awal (lower bound) patch."""
        lo = [min(0, a - s) for a, s in zip(shape, self.patch)]   # negatif kalau volume < patch
        hi = [max(0, a - s) for a, s in zip(shape, self.patch)]
        if force_fg and len(locs) > 0:
            k = rng.choice(list(locs.keys()))                      # pilih kelas acak yang ada
            vox = locs[k][rng.integers(len(locs[k]))]              # pilih voxel acak dari kelas itu
            lb = [int(v - s // 2) for v, s in zip(vox, self.patch)]  # pusatkan patch di voxel
            return [int(np.clip(l, a, b)) for l, a, b in zip(lb, lo, hi)]
        return [int(rng.integers(a, b + 1)) for a, b in zip(lo, hi)]

    # ---------- augmentasi ----------
    def _spatial(self, img, seg, rng):
        """Rotasi di bidang (y, x) dan scaling, lewat satu transformasi affine."""
        do_rot, do_scale = rng.random() < 0.2, rng.random() < 0.2
        if not (do_rot or do_scale):
            return img, seg
        theta = np.deg2rad(rng.uniform(-30, 30)) if do_rot else 0.0
        scale = rng.uniform(0.7, 1.4) if do_scale else 1.0
        c, s = np.cos(theta), np.sin(theta)
        M = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        M[0, 0] = 1.0
        M = M / scale                      # scale > 1 = memperbesar objek
        M[0, 0] = 1.0 / scale
        center = np.array(img.shape) / 2.0
        offset = center - M @ center       # putar/skala terhadap pusat patch
        img = affine_transform(img, M, offset=offset, order=1, mode="constant", cval=0.0)
        seg = affine_transform(seg, M, offset=offset, order=0, mode="constant", cval=0)
        return img.astype(np.float32), seg.astype(np.uint8)

    def _intensity(self, img, rng):
        if rng.random() < 0.1:                                   # noise Gaussian
            img = img + rng.normal(0, np.sqrt(rng.uniform(0, 0.1)), img.shape)
        if rng.random() < 0.2:                                   # blur
            img = gaussian_filter(img, sigma=rng.uniform(0.5, 1.0))
        if rng.random() < 0.15:                                  # brightness
            img = img * rng.uniform(0.75, 1.25)
        if rng.random() < 0.15:                                  # contrast
            m, lo, hi = img.mean(), img.min(), img.max()
            img = np.clip((img - m) * rng.uniform(0.75, 1.25) + m, lo, hi)
        if rng.random() < 0.3:                                   # gamma
            lo, hi = img.min(), img.max()
            norm = (img - lo) / (hi - lo + 1e-8)
            img = norm ** rng.uniform(0.7, 1.5) * (hi - lo) + lo
        return img.astype(np.float32)

    def _mirror(self, img, seg, rng):
        for ax in range(3):
            if rng.random() < 0.5:
                img, seg = np.flip(img, ax), np.flip(seg, ax)
        return np.ascontiguousarray(img), np.ascontiguousarray(seg)

    # ---------- satu sampel ----------
    def __getitem__(self, idx):
        # Tiap worker/sampel butuh generator acak sendiri, kalau tidak hasilnya bisa kembar
        rng = np.random.default_rng(None if self.seed is None else self.seed + idx)
        i = int(rng.integers(len(self.case_ids)))
        img_full, seg_full = self.imgs[i], self.segs[i]

        force_fg = rng.random() < self.fg_prob
        lb = self._sample_lb(img_full.shape, self.fg_locs[i], rng, force_fg)
        img = _crop(img_full, lb, self.patch).astype(np.float32)
        seg = _crop(seg_full, lb, self.patch).astype(np.uint8)

        if self.augment:
            img, seg = self._spatial(img, seg, rng)
            img = self._intensity(img, rng)
            img, seg = self._mirror(img, seg, rng)

        return torch.from_numpy(img)[None], torch.from_numpy(seg.astype(np.int64))


if __name__ == "__main__":
    DATA_DIR = "../preproc"

    print("[1/3] Memuat dataset...")
    ds = LiverPatchDataset(DATA_DIR)
    print(f"      {len(ds.case_ids)} kasus, patch {ds.patch}")

    print("[2/3] Mengambil 50 batch...")
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4)
    n_total = n_tumor = 0
    for step, (x, y) in enumerate(dl):
        if step == 0:
            print(f"      x: {tuple(x.shape)} {x.dtype} | y: {tuple(y.shape)} {y.dtype} | label: {y.unique().tolist()}")
        n_total += x.shape[0]
        n_tumor += int(((y == 2).flatten(1).any(1)).sum())
        if (step + 1) % 10 == 0:
            print(f"      batch {step + 1}/50")
        if step + 1 == 50:
            break

    print("[3/3] Ringkasan")
    print(f"      patch berisi tumor: {n_tumor}/{n_total} ({n_tumor / n_total:.0%})")


"""
Check Augumentasi
"""
# import numpy as np
# import matplotlib.pyplot as plt
# from dataset import LiverPatchDataset

# ds = LiverPatchDataset("../preproc", seed=0)   # seed tetap: patch dasarnya sama untuk aug on/off
# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# for j in range(4):
#     for row, aug in enumerate([False, True]):
#         ds.augment = aug
#         x, y = ds[j]
#         x, y = x[0].numpy(), y.numpy()
#         z = int(np.argmax((y > 0).sum(axis=(1, 2))))
#         ax = axes[row, j]
#         ax.imshow(x[z], cmap="gray")
#         ax.imshow(np.ma.masked_equal(y[z], 0), alpha=0.5, cmap="autumn", vmin=0, vmax=2)
#         ax.set_title(f"sampel {j} | augment={aug}", fontsize=9)
#         ax.axis("off")
# plt.tight_layout()
# plt.savefig("check_augment.png", dpi=110)


