import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def load(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)

def overlay(ax, img, seg, z, title):
    ax.imshow(img[z], cmap="gray")
    ax.imshow(np.ma.masked_equal(seg[z], 0), alpha=0.5, cmap="autumn", vmin=0, vmax=2)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def check_case(case_id, raw_img, raw_seg, prep_dir="preproc"):
    # Sebelum
    img0, seg0 = load(raw_img), load(raw_seg)
    # Sesudah
    img1 = np.load(f"{prep_dir}/{case_id}_img.npy")
    seg1 = np.load(f"{prep_dir}/{case_id}_seg.npy")

    # Pilih slice dengan area tumor terbesar (kalau tidak ada, liver terbesar)
    def best_slice(seg):
        t = (seg == 2).sum(axis=(1, 2))
        return int(np.argmax(t)) if t.max() > 0 else int(np.argmax((seg > 0).sum(axis=(1, 2))))

    z0, z1 = best_slice(seg0), best_slice(seg1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    overlay(ax[0], img0, seg0, z0, f"{case_id} sebelum {img0.shape}, slice {z0}")
    overlay(ax[1], img1, seg1, z1, f"{case_id} sesudah {img1.shape}, slice {z1}")
    plt.tight_layout()
    plt.savefig(f"check_{case_id}.png", dpi=120)
    plt.close()

# Contoh: im3 dan im27 (spacing paling ekstrem), plus im0
check_case("im3",  "train/imagesTr/im3.nii.gz",  "train/labelsTr/lb3.nii.gz")
check_case("im27", "train/imagesTr/im27.nii.gz", "train/labelsTr/lb27.nii.gz")
check_case("im0",  "train/imagesTr/im0.nii.gz",  "train/labelsTr/lb0.nii.gz")