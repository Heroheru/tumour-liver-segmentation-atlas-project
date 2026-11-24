import sys, json
sys.path.append("/content/drive/MyDrive/Project_Atlas_Individual")      # sesuaikan dengan lokasi folder proyekmu
sys.path.append("/content/drive/MyDrive/Project_Atlas_Individual/modelling")
import torch
from Data_Processing.dataset import LiverPatchDataset
from model import UNet3D

RUN = "/content/drive/MyDrive/Project_Atlas_Individual/runs/percobaan2"   # sesuaikan dengan --out-mu
DATA = "/content/preproc"
device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(f"{RUN}/best.pt", map_location="cpu")
split = json.load(open(f"{RUN}/split.json"))
ds = LiverPatchDataset(DATA, patch_size=ck["patch"], augment=False,
                       case_ids=split["val"], fg_prob=1.0, seed=123)

model = UNet3D(3)
model.load_state_dict(ck["model"])
model.eval().to(device)

print(f"checkpoint dari epoch {ck['epoch'] + 1}")
n_shown = 0
for i in range(30):
    x, y = ds[i]
    gt_t = y == 2
    if not gt_t.any():
        continue                                             # lewati patch tanpa tumor
    with torch.no_grad():
        p = torch.softmax(model(x[None].to(device)), dim=1)[0].cpu()   # (3, D, H, W)
    pred = p.argmax(0)
    inside = p[2][gt_t].mean().item()
    outside = p[2][~gt_t].mean().item()
    print(f"patch {i:>2} | tumor GT: {int(gt_t.sum()):>7} vox | prediksi tumor: {int((pred == 2).sum()):>7} vox | "
          f"P(tumor) dalam GT {inside:.3f} vs di luar {outside:.3f}")
    n_shown += 1
    if n_shown == 8:
        break