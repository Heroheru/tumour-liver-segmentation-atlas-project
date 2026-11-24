""" 
Pada bagian ini merupakan kode dimana data di preprocessing terlebih dahulu dengan urutan :
1. crop non-zero 
2. resample 
3. z-score
4. simpan.

"""
import argparse
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

#Definisi target spacing
# Dari nnUNetPlans.json (3d_fullres), urutan (z, y, x)

"""
=========Pengetahuan=========
Dalam data medis seperti NIFTI, TARGET_SPACING menyatakan jarak spasial antar voxel (pixel 3D), dan satuannya hampir selalu dalam milimeter (mm).
Berikut adalah penjelasan detail mengenai maksud dari nilai [3.0, 1.0416667, 1.0416667]:

1. Arti Nilai Spacing (Z, Y, X)

Format spacing biasanya diurutkan berdasarkan sumbu \((Z, Y, X)\) atau \((X, Y, Z)\) tergantung pada orientasi data (misalnya axial, sagittal, atau coronal). Pada contoh Anda, urutannya umumnya adalah \((Z, Y, X)\):
• Z = 3.0 (Slice Thickness): Artinya ketebalan setiap slice (irisan) adalah 3 mm. Jarak dari pusat satu slice ke pusat slice berikutnya secara vertikal adalah 3 mm.
• Y = 1.0416667: Artinya resolusi piksel dalam satu slice secara vertikal (baris) adalah sekitar 1.04 mm.
• X = 1.0416667: Artinya resolusi piksel dalam satu slice secara horizontal (kolom) adalah sekitar 1.04 mm.

2. Ilustrasi Fisik Voxel

Jika dibayangkan sebagai kotak 3D (voxel):
• Ukuran kotak tersebut adalah 3.0 mm × 1.04 mm × 1.04 mm.
• Kotak ini berbentuk balok (tidak simetris/anisotropik) karena ketebalan slice (\(Z\)) jauh lebih besar daripada resolusi bidangnya (\(X\) dan \(Y\)).

"""
TARGET_SPACING = np.array([3.0, 1.0416667, 1.0416667]) #Z,Y,X

#FUngsi Loda Image .NTI
def load_nifti(path):
    """Baca file NIfTI -> array (z, y, x) + spacing (z, y, x)."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)           # sudah urutan (z, y, x)
    spacing = np.array(img.GetSpacing()[::-1])  # SimpleITK memberi (x, y, z) -> dibalik
    return arr, spacing

def crop_nonzero(img, seg):
    """Buang tepi yang seluruhnya bernilai 0 (biasanya padding)."""
    coords = np.argwhere(img != 0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1               # +1 karena slice eksklusif di ujung
    bbox = tuple(slice(int(a), int(b)) for a, b in zip(mins, maxs))
    return img[bbox], seg[bbox], np.stack([mins, maxs])  # bbox disimpan untuk inverse nanti

def resample(arr, old_spacing, new_spacing, is_seg):
    """
    Ubah ukuran array supaya voxel-nya berspacing new_spacing.
    Ukuran baru = ukuran lama * spacing lama / spacing baru
    (contoh: z 80 slice @ 3.0mm ke target 3.0mm -> tetap 80; XY 0.78mm -> 1.04mm mengecil).
    """
    new_shape = np.round(np.array(arr.shape) * old_spacing / new_spacing).astype(int)
    factors = new_shape / np.array(arr.shape)
    if is_seg:
        # order 0 = nearest neighbor: label tetap bilangan bulat (0/1/2), tidak jadi 0.5
        out = zoom(arr, factors, order=0, mode="nearest")
        return out.astype(np.uint8)
    # order 3 = cubic spline: halus untuk citra intensitas
    out = zoom(arr.astype(np.float32), factors, order=3, mode="nearest")
    return out.astype(np.float32)

def zscore(img):
    """Normalisasi per citra: rata-rata 0, std 1. MRI tidak punya skala absolut."""
    return ((img - img.mean()) / (img.std() + 1e-8)).astype(np.float32)

def preprocess_case(img_path, seg_path, out_dir, case_id):
    img, spacing = load_nifti(img_path)
    seg, _ = load_nifti(seg_path)
    img = img.astype(np.float32)
    seg = seg.astype(np.uint8)
    assert img.shape == seg.shape, f"{case_id}: shape citra {img.shape} != mask {seg.shape}"

    original_shape = np.array(img.shape)

    img, seg, bbox = crop_nonzero(img, seg)
    img = resample(img, spacing, TARGET_SPACING, is_seg=False)
    seg = resample(seg, spacing, TARGET_SPACING, is_seg=True)
    img = zscore(img)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{case_id}_img.npy", img)
    np.save(out_dir / f"{case_id}_seg.npy", seg)
    # Metadata untuk mengembalikan prediksi ke ruang asli saat evaluasi
    np.savez(out_dir / f"{case_id}_meta.npz",
             spacing=spacing, original_shape=original_shape, bbox=bbox)

    print(f"{case_id}: {tuple(original_shape)} @ {np.round(spacing, 2)} -> "
          f"{img.shape} | label: {np.unique(seg)} | mean={img.mean():.3f} std={img.std():.3f}")

def find_label(img_path, labels_dir, img_prefix="im", lbl_prefix="lb"):
    """im0.nii.gz -> lb0.nii.gz (ganti awalan 'im' dengan 'lb')."""
    name = img_path.name
    if not name.startswith(img_prefix):
        raise ValueError(f"{name} tidak diawali '{img_prefix}'")
    lbl_name = lbl_prefix + name[len(img_prefix):]
    lbl_path = Path(labels_dir) / lbl_name
    if not lbl_path.exists():
        raise FileNotFoundError(f"Mask untuk {name} tidak ditemukan: {lbl_path}")
    case_id = name.split(".nii")[0]          # contoh: "im0"
    return lbl_path, case_id

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Urutkan secara numerik (im2 sebelum im10), bukan alfabet
    def num(p):
        return int(re.search(r"\d+", p.name).group())

    for img_path in sorted(Path(args.images).glob("im*.nii*"), key=num):
        seg_path, case_id = find_label(img_path, args.labels)
        preprocess_case(img_path, seg_path, args.out, case_id)

        """
        load_nifti: jebakan terbesar ada di sini. SimpleITK mengembalikan spacing (x, y, z), sedangkan array numpy-nya (z, y, x). Kalau spacing tidak dibalik, resampling akan memakai spacing yang salah di tiap sumbu dan hasilnya terlihat “gepeng” atau melar.

        crop_nonzero: argwhere(img != 0) mengambil koordinat semua voxel non-nol, lalu min dan max per sumbu menjadi kotak pembatas. Bbox disimpan karena saat evaluasi prediksimu harus ditempel kembali ke ukuran asli, dan tanpa bbox kamu tidak tahu posisinya.

        resample: rumusnya menjaga ukuran fisik. Volume 80 slice × 3 mm = 240 mm tebal. Kalau target spacing z sama (3,0), jumlah slice tetap. Di XY, voxel 0,78 mm yang dibesarkan ke 1,04 mm membuat matriksnya mengecil sekitar 25%. Interpolasi untuk citra memakai cubic (order 3), untuk mask nearest neighbor (order 0), supaya label tidak berubah menjadi nilai pecahan.

        zscore: dihitung setelah resample, pada seluruh volume. Ini penting karena MRI tiap scanner punya skala intensitas berbeda. Tanpa ini, model akan “melihat” perbedaan scanner, bukan perbedaan jaringan. Epsilon 1e-8 mencegah pembagian nol.

        Metadata: spacing asli dan shape asli wajib disimpan. Metrik jarak (surface Dice, ASSD, HD) harus dihitung di ruang asli dengan spacing asli

        """


        """
        im0: (88, 368, 512) @ [2.3 0.78 0.78] -> (67, 276, 383)
      ukuran asli       spacing asli       ukuran setelah resample
        
        Ukuran asli (z, y, x) dalam voxel, yaitu jumlah slice dan ukuran matriks per slice. Ini dicetak sebelum crop, jadi angkanya bisa beda 1 voxel dengan ukuran yang masuk ke resample (misal x 512 jadi 511 setelah crop).
        Spacing asli (z, y, x) dalam mm. Kolom y dan x selalu sama di datamu, jadi voxel in-plane-nya persegi.
        Ukuran setelah resample (z, y, x), yaitu ukuran citra setelah crop dan resample ke spacing seragam 3,0 × 1,04 × 1,04 mm. Inilah ukuran yang akan dilihat model.

        Bagaimana ukuran berubah mengikuti spacing:

        Spacing lebih halus dari target (misal 0,78 mm), ukuran mengecil. Contoh im0: x 511 → 383, dan z 88 → 67 karena z asli 2,3 mm.
        Spacing lebih kasar dari target (misal z 3,5 atau 4,0 mm, atau XY 1,25 mm), ukuran membesar. Contoh im18: z 72 → 84, dan im3: z 44 → 59. Ini hanya interpolasi, tidak ada informasi baru.
        Spacing sudah sama dengan target (im2, im9, im38), ukuran hampir tidak berubah.

        Beberapa hal dari log-mu:

        Semua 60 kasus berhasil dan label-nya selalu [0 1 2]. Mean dan std juga benar di semuanya.
        Ukuran akhir bervariasi: z antara 48 (im27) sampai 100 (im47), y antara 231 sampai 367, x antara 335 sampai 479. Variasi ini wajar dan tidak masalah, karena model dilatih dengan patch tetap, bukan seluruh volume.
        Semua kasus lebih besar atau sama dengan patch 48 × 192 × 224. Yang paling mepet adalah im27 dengan z = 48 (pas sama patch). Jadi di datamu padding praktis tidak terpakai, tapi tetap kita tulis di data loader supaya aman untuk data test.
        Ada yang _cases_ 80 slice dengan z tetap 80 sementara y/x berubah, misal im7: (80, 276, 352) → (80, 301, 383). Itu karena z-nya sudah 3,0 mm, jadi hanya XY yang diresample. Normal.
        """