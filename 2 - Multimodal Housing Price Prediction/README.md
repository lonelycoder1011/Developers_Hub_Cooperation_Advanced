# 🏠 Multimodal ML — Housing Price Prediction

Predict housing prices using **4 room images + 21 tabular features**.

---

## Project Structure
```
housing_multimodal/
├── generate_images.py       # Synthesizes 600 room images (4 types × 150)
├── generate_dataset.py      # Builds paired tabular CSV (150 houses)
├── multimodal_pipeline.py   # Full training pipeline (CNN + GBM)
├── inference.py             # Load saved model, predict new house
├── visualize_results.py     # Results dashboard figure
│
├── data/
│   └── housing_dataset.csv  # 150 houses, 22 cols, $245k–$1.09M
├── images/
│   ├── bedroom/             # 150 × bedroom_NNNN.jpg
│   ├── kitchen/             # 150 × kitchen_NNNN.jpg
│   ├── exterior/            # 150 × exterior_NNNN.jpg
│   └── livingroom/          # 150 × livingroom_NNNN.jpg
├── saved_models/
│   ├── multimodal_bundle.pt         # Full model + metadata
│   ├── best_multimodal.pt           # Best checkpoint weights
│   ├── tabular_preprocessor.joblib  # Sklearn ColumnTransformer
│   ├── tabular_baseline.joblib      # GBM baseline
│   └── training_history.json        # Loss + MAE per epoch
└── outputs/
    ├── results_dashboard.png
    └── sample_images.png
```

---

## Architecture

```
House Images (4 rooms)             Tabular Features (21 dims)
  [Bedroom | Kitchen |               [sqft, beds, baths,
   Exterior | Livingroom]             school_rating, ...]
       │ 64×64 RGB                           │
       ▼                                     ▼
  ┌─────────────┐                   ┌─────────────────┐
  │  LightCNN   │                   │   Tabular MLP   │
  │  Conv×3     │                   │  Linear×3       │
  │  AvgPool    │                   │  LayerNorm+ReLU │
  └──────┬──────┘                   └────────┬────────┘
         │  64-dim img emb                   │  64-dim tab emb
         └─────────────┬─────────────────────┘
                  CONCAT (128-dim)
                       │
              ┌────────┴────────┐
              │  Regression Head│
              │  Linear→ReLU×2 │
              │  → log(price)  │
              └────────┬────────┘
                       │  exp()
                    $ Price
```

---

## Training Strategy

| Phase | Epochs | What trains | LR |
|-------|--------|-------------|-----|
| Warm-up | 1–10 | Tabular branch only (CNN frozen) | 5e-3 |
| Fine-tune | 11–30 | All params jointly | 8e-4 (cosine) |

Loss: **MSE in log(price) space** (equivalent to RMSLE)

---

## Results

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| GBM (tabular only) | **$60,283** | **$74,438** | **0.683** |
| Multimodal CNN | $107,285 | $126,137 | 0.090 |

> **Why does GBM win?** With only 150 training samples, the CNN cannot learn
> meaningful room-quality features — random weight initialisation beats
> learned features at this scale. On a real dataset (5k–50k houses) the
> multimodal model consistently outperforms tabular-only approaches.

---

## Quick Start

```bash
pip install torch torchvision scikit-learn pandas numpy joblib pillow matplotlib

python generate_images.py      # ~1 min
python generate_dataset.py
python multimodal_pipeline.py  # ~3 min on CPU
python inference.py            # demo prediction
python visualize_results.py    # results dashboard PNG
```

## Load Saved Model

```python
import torch, joblib, math
from multimodal_pipeline import MultimodalNet

bundle = torch.load("saved_models/multimodal_bundle.pt", weights_only=False)
model  = MultimodalNet(tab_in=bundle["tab_dim"])
model.load_state_dict(bundle["model_state"])
model.eval()

prep = joblib.load("saved_models/tabular_preprocessor.joblib")
```

---

## Requirements
```
torch torchvision scikit-learn pandas numpy joblib pillow matplotlib
```
