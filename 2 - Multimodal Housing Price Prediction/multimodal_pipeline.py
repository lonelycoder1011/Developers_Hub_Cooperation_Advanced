"""
===============================================================
  Multimodal ML — Housing Price Prediction
  CNN (image features)  +  Tabular data  →  Price
===============================================================
  Architecture:
    • LightCNN extracts quality signal from 4 room images
    • Tabular MLP handles structured features
    • Late-fusion → regression head → log(price)
  Training strategy:
    • Phase 1 (10 ep): tabular branch only (CNN frozen)
    • Phase 2 (20 ep): full joint fine-tuning
  Metrics: MAE, RMSE, R²
"""

import warnings; warnings.filterwarnings("ignore")
import json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.metrics         import mean_absolute_error
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.ensemble        import GradientBoostingRegressor

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "housing_dataset.csv"
MODEL_DIR = BASE_DIR / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEVICE] {DEVICE}")

IMG_ROOMS    = ["img_bedroom", "img_kitchen", "img_exterior", "img_livingroom"]
NUMERIC_COLS = [
    "sqft","bedrooms","bathrooms","garage_cars","year_built","stories",
    "lot_sqft","school_rating","dist_downtown","crime_index",
    "has_pool","has_basement","renovation_yr","avg_img_quality",
]
CAT_COLS = ["neighborhood","condition"]

# ── Transforms ──────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
VAL_TF = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════
class HousingDataset(Dataset):
    def __init__(self, df, tab_arr, transform=None):
        self.df        = df.reset_index(drop=True)
        self.tab       = tab_arr.astype(np.float32)
        self.tf        = transform or VAL_TF
        self.log_price = np.log(df["price"].values.astype(np.float32))

    def __len__(self): return len(self.df)

    def _img(self, path):
        try:    return Image.open(path).convert("RGB")
        except: return Image.new("RGB",(64,64),(110,110,110))

    def __getitem__(self, i):
        row  = self.df.iloc[i]
        imgs = torch.stack([self.tf(self._img(row[r])) for r in IMG_ROOMS])
        tab  = torch.from_numpy(self.tab[i])
        tgt  = torch.tensor(self.log_price[i])
        return imgs, tab, tgt


# ══════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════
class LightCNN(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(), nn.MaxPool2d(2),   # 16
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(), nn.MaxPool2d(2),   # 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, out)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class ImageBranch(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.cnn = LightCNN(out=dim)
        self.dim = dim

    def forward(self, imgs):
        B,N,C,H,W = imgs.shape
        f = self.cnn(imgs.view(B*N,C,H,W))
        return f.view(B,N,self.dim).mean(1)


class TabBranch(nn.Module):
    def __init__(self, in_dim, out=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64),     nn.LayerNorm(64),  nn.ReLU(),
            nn.Linear(64, out),    nn.ReLU(),
        )
    def forward(self, x): return self.net(x)


class MultimodalNet(nn.Module):
    def __init__(self, tab_in):
        super().__init__()
        self.img = ImageBranch(dim=64)
        self.tab = TabBranch(tab_in, out=64)
        self.head = nn.Sequential(
            nn.Linear(128,64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64,32),  nn.ReLU(),
            nn.Linear(32,1),
        )

    def forward(self, imgs, tab):
        return self.head(torch.cat([self.img(imgs), self.tab(tab)], 1)).squeeze(1)

    def freeze_cnn(self):
        for p in self.img.parameters(): p.requires_grad = False

    def unfreeze_cnn(self):
        for p in self.img.parameters(): p.requires_grad = True


# ══════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════════
def run_epoch(model, loader, opt, crit, train=True):
    model.train(train)
    total, ps, ts = 0., [], []
    with (torch.enable_grad() if train else torch.no_grad()):
        for imgs, tab, tgt in loader:
            imgs,tab,tgt = imgs.to(DEVICE), tab.to(DEVICE), tgt.to(DEVICE)
            if train: opt.zero_grad()
            out  = model(imgs, tab)
            loss = crit(out, tgt)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(tgt)
            ps.append(out.detach().cpu().numpy())
            ts.append(tgt.cpu().numpy())
    return total/len(loader.dataset), np.concatenate(ps), np.concatenate(ts)


def get_metrics(lp, lt):
    p,t = np.exp(lp), np.exp(lt)
    mae  = mean_absolute_error(t,p)
    rmse = math.sqrt(np.mean((p-t)**2))
    ss_r = np.sum((t-p)**2)
    ss_t = np.sum((t-t.mean())**2)
    r2   = 1 - ss_r/ss_t if ss_t else float("nan")
    return dict(MAE=mae, RMSE=rmse, R2=r2, MAE_pct=mae/t.mean()*100)


def show_metrics(m, tag):
    bar = "═"*54
    print(f"\n{bar}")
    print(f"  {tag}")
    print(bar)
    print(f"  MAE   : ${m['MAE']:>13,.0f}   ({m['MAE_pct']:.1f}% of mean price)")
    print(f"  RMSE  : ${m['RMSE']:>13,.0f}")
    print(f"  R²    : {m['R2']:>15.4f}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("="*55)
    print("  Multimodal Housing Price Prediction Pipeline")
    print("="*55)

    # ── Data ────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"\n[DATA] {len(df)} houses | "
          f"price ${df.price.min():,.0f} – ${df.price.max():,.0f} "
          f"| median ${df.price.median():,.0f}")

    df_tr, df_tmp = train_test_split(df, test_size=0.30, random_state=42)
    df_val, df_te = train_test_split(df_tmp, test_size=0.50, random_state=42)
    print(f"[SPLIT] train={len(df_tr)}  val={len(df_val)}  test={len(df_te)}")

    # ── Tabular preprocessing ────────────────────────────────────────
    prep = ColumnTransformer([
        ("num", Pipeline([("sc", StandardScaler())]),                             NUMERIC_COLS),
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore",
                                                sparse_output=False))]), CAT_COLS),
    ])
    X_tr  = prep.fit_transform(df_tr)
    X_val = prep.transform(df_val)
    X_te  = prep.transform(df_te)
    tab_dim = X_tr.shape[1]
    print(f"[PREP] Tabular dim after encoding: {tab_dim}")
    joblib.dump(prep, MODEL_DIR / "tabular_preprocessor.joblib")

    # ── Tabular-only GBM baseline ───────────────────────────────────
    print("\n[BASELINE] Gradient Boosting on tabular features …")
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42)
    gbm.fit(X_tr, np.log(df_tr["price"].values))
    gbm_pred = gbm.predict(X_te)
    base_m   = get_metrics(gbm_pred, np.log(df_te["price"].values))
    show_metrics(base_m, "Tabular-Only GBM — Test Set")
    joblib.dump(gbm, MODEL_DIR / "tabular_baseline.joblib")

    # ── Dataloaders ──────────────────────────────────────────────────
    ds_tr  = HousingDataset(df_tr.reset_index(drop=True),  X_tr,  TRAIN_TF)
    ds_val = HousingDataset(df_val.reset_index(drop=True), X_val, VAL_TF)
    ds_te  = HousingDataset(df_te.reset_index(drop=True),  X_te,  VAL_TF)
    kw = dict(num_workers=0, pin_memory=False)
    dl_tr  = DataLoader(ds_tr,  batch_size=16, shuffle=True,  **kw)
    dl_val = DataLoader(ds_val, batch_size=16, shuffle=False, **kw)
    dl_te  = DataLoader(ds_te,  batch_size=16, shuffle=False, **kw)

    # ── Model ─────────────────────────────────────────────────────────
    model = MultimodalNet(tab_in=tab_dim).to(DEVICE)
    crit  = nn.MSELoss()
    best_val = float("inf")
    hist = dict(train=[], val=[], val_mae=[])

    # ────────────────────────────────────────────────────────────────
    # Phase 1 — tabular warm-up (CNN frozen, 10 epochs)
    # ────────────────────────────────────────────────────────────────
    print("\n[PHASE 1] Tabular warm-up — CNN frozen (10 epochs)")
    print("─"*54)
    model.freeze_cnn()
    opt1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=5e-3, weight_decay=1e-4)
    sch1 = optim.lr_scheduler.StepLR(opt1, step_size=4, gamma=0.5)

    for ep in range(1, 11):
        tl, _, _    = run_epoch(model, dl_tr,  opt1, crit, True)
        vl, vp, vt  = run_epoch(model, dl_val, opt1, crit, False)
        vm = get_metrics(vp, vt)
        sch1.step()
        hist["train"].append(tl); hist["val"].append(vl)
        hist["val_mae"].append(vm["MAE"])
        marker = ""
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), MODEL_DIR/"best_multimodal.pt")
            marker = " ✓"
        print(f"  ep {ep:02d}/10  train={tl:.4f}  val={vl:.4f}  "
              f"MAE=${vm['MAE']:,.0f}  R²={vm['R2']:.3f}{marker}")

    # ────────────────────────────────────────────────────────────────
    # Phase 2 — joint fine-tuning (all params, 20 epochs)
    # ────────────────────────────────────────────────────────────────
    print("\n[PHASE 2] Joint fine-tune — all params (20 epochs)")
    print("─"*54)
    model.unfreeze_cnn()
    opt2 = optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-4)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=20, eta_min=1e-5)

    for ep in range(1, 21):
        tl, _, _    = run_epoch(model, dl_tr,  opt2, crit, True)
        vl, vp, vt  = run_epoch(model, dl_val, opt2, crit, False)
        vm = get_metrics(vp, vt)
        sch2.step()
        hist["train"].append(tl); hist["val"].append(vl)
        hist["val_mae"].append(vm["MAE"])
        marker = ""
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), MODEL_DIR/"best_multimodal.pt")
            marker = " ✓"
        print(f"  ep {ep:02d}/20  train={tl:.4f}  val={vl:.4f}  "
              f"MAE=${vm['MAE']:,.0f}  R²={vm['R2']:.3f}{marker}")

    # ── Evaluate on test set ─────────────────────────────────────────
    model.load_state_dict(torch.load(MODEL_DIR/"best_multimodal.pt", map_location=DEVICE))
    _, te_p, te_t = run_epoch(model, dl_te, opt2, crit, False)
    multi_m = get_metrics(te_p, te_t)
    show_metrics(multi_m, "Multimodal CNN+Tabular — Test Set")

    # ── Head-to-head comparison ──────────────────────────────────────
    print("\n" + "═"*60)
    print("  Final Comparison — Test Set")
    print("═"*60)
    print(f"  {'Metric':<8}  {'GBM (tabular only)':>20}  {'Multimodal CNN':>16}  Winner")
    print(f"  {'─'*8}  {'─'*20}  {'─'*16}  {'─'*6}")
    for k in ["MAE","RMSE","R2"]:
        bv, mv = base_m[k], multi_m[k]
        better = (mv < bv) if k != "R2" else (mv > bv)
        w = "Multi 🏆" if better else "GBM 🏆"
        if k != "R2":
            print(f"  {k:<8}  ${bv:>19,.0f}  ${mv:>15,.0f}  {w}")
        else:
            print(f"  {k:<8}  {bv:>20.4f}  {mv:>16.4f}  {w}")
    print()

    # ── Save everything ──────────────────────────────────────────────
    torch.save(dict(
        model_state  = model.state_dict(),
        tab_dim      = tab_dim,
        img_rooms    = IMG_ROOMS,
        test_metrics = multi_m,
        base_metrics = base_m,
        history      = hist,
    ), MODEL_DIR / "multimodal_bundle.pt")

    with open(MODEL_DIR / "training_history.json", "w") as f:
        json.dump(hist, f, indent=2)

    print("[SAVE] multimodal_bundle.pt")
    print("[SAVE] tabular_preprocessor.joblib")
    print("[SAVE] tabular_baseline.joblib")
    print("[SAVE] best_multimodal.pt  (weights only)")
    print("\n✅  Pipeline complete!")
    return multi_m, base_m, hist

if __name__ == "__main__":
    main()
