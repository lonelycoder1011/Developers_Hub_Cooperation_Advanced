"""
inference.py
─────────────────────────────────────────────────────────
Load the saved multimodal pipeline and predict on a new house.
Usage:
    python inference.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib
from PIL import Image
from torchvision import transforms

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "saved_models"
DEVICE    = torch.device("cpu")

IMG_ROOMS    = ["img_bedroom","img_kitchen","img_exterior","img_livingroom"]
NUMERIC_COLS = [
    "sqft","bedrooms","bathrooms","garage_cars","year_built","stories",
    "lot_sqft","school_rating","dist_downtown","crime_index",
    "has_pool","has_basement","renovation_yr","avg_img_quality",
]
CAT_COLS = ["neighborhood","condition"]

VAL_TF = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def load_bundle():
    from multimodal_pipeline import MultimodalNet
    bundle = torch.load(MODEL_DIR / "multimodal_bundle.pt", map_location=DEVICE, weights_only=False)
    model  = MultimodalNet(tab_in=bundle["tab_dim"]).to(DEVICE)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    prep = joblib.load(MODEL_DIR / "tabular_preprocessor.joblib")
    print("✅ Models loaded.")
    return model, prep, bundle


def predict(house: dict, img_paths: dict, model, prep) -> float:
    """
    house     : raw tabular features dict
    img_paths : {room: path_string}  — use any JPEG
    Returns   : predicted price in dollars
    """
    df  = pd.DataFrame([house])
    X   = prep.transform(df).astype(np.float32)
    tab = torch.from_numpy(X).to(DEVICE)

    imgs = []
    for room in IMG_ROOMS:
        key  = room.replace("img_","")
        path = img_paths.get(key)
        try:    img = Image.open(path).convert("RGB") if path else None
        except: img = None
        if img is None: img = Image.new("RGB",(64,64),(110,110,110))
        imgs.append(VAL_TF(img))
    imgs_t = torch.stack(imgs).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        log_p = model(imgs_t, tab).item()
    return math.exp(log_p)


if __name__ == "__main__":
    model, prep, bundle = load_bundle()

    # ── Example house ─────────────────────────────────────────────
    sample = {
        "sqft": 2200, "bedrooms": 3, "bathrooms": 2.0,
        "garage_cars": 2, "year_built": 2005, "stories": 2,
        "lot_sqft": 6000, "neighborhood": "suburban",
        "school_rating": 7.5, "dist_downtown": 8.0, "crime_index": 4.0,
        "condition": "good", "has_pool": 0, "has_basement": 1,
        "renovation_yr": 2018, "avg_img_quality": 0.72,
    }
    imgs = {
        "bedroom":    str(BASE_DIR/"images"/"bedroom"   /"bedroom_0000.jpg"),
        "kitchen":    str(BASE_DIR/"images"/"kitchen"   /"kitchen_0000.jpg"),
        "exterior":   str(BASE_DIR/"images"/"exterior"  /"exterior_0000.jpg"),
        "livingroom": str(BASE_DIR/"images"/"livingroom"/"livingroom_0000.jpg"),
    }

    price = predict(sample, imgs, model, prep)
    print(f"\n🏠 Predicted price : ${price:,.0f}")

    m = bundle["test_metrics"]
    b = bundle["base_metrics"]
    print(f"\n📊 Test Metrics")
    print(f"   {'Model':<22}  {'MAE':>10}  {'RMSE':>10}  {'R²':>8}")
    print(f"   {'─'*22}  {'─'*10}  {'─'*10}  {'─'*8}")
    print(f"   {'GBM (tabular only)':<22}  ${b['MAE']:>9,.0f}  ${b['RMSE']:>9,.0f}  {b['R2']:>8.4f}")
    print(f"   {'Multimodal CNN':<22}  ${m['MAE']:>9,.0f}  ${m['RMSE']:>9,.0f}  {m['R2']:>8.4f}")
