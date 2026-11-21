import os
# Suppress oneDNN and verbose TF logging warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path
from typing import Optional
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import requests

# =============================================================================
# Configuration & Calibrated Thresholds
# =============================================================================
OOD_THRESHOLD = float(os.environ.get("OOD_THRESHOLD", "0.45"))
OOD_BORDERLINE = float(os.environ.get("OOD_BORDERLINE", "0.08"))
MIN_CONFIDENCE = float(os.environ.get("MIN_CONFIDENCE", "0.40"))
PLANTNET_MIN_SCORE = float(os.environ.get("PLANTNET_MIN_SCORE", "0.80"))

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
OOD_STATS_PATH = BASE_DIR / "model" / "ood_stats.npz"

app = FastAPI(title="AgriSL Disease Detection ML Service", version="2.2.0")

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AgriSL ML Service",
        "model_loaded": model is not None,
        "ood_centroids_loaded": OOD_CENTROIDS is not None,
    }

# Load classification model (raw 0-255 RGB, internal normalization)
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

IMG_SIZE = (224, 224)

# =============================================================================
# Extract Embedding Layer Dynamically (GlobalAveragePooling2D)
# =============================================================================
def find_pooling_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return layer
    for layer in reversed(m.layers):
        if "pool" in layer.name.lower():
            return layer
    return None

pool_layer = find_pooling_layer(model)
feature_model = None
if pool_layer is not None:
    try:
        feature_model = tf.keras.Model(inputs=model.input, outputs=pool_layer.output)
        print(f"[ML Init] Feature model built using pooling layer '{pool_layer.name}'")
    except Exception as e:
        print(f"[ML Init WARNING] Could not construct feature model: {e}")
else:
    print("[ML Init WARNING] GlobalAveragePooling2D layer not found.")

# =============================================================================
# Load 35 Per-Class Centroids for OOD Similarity
# =============================================================================
OOD_CENTROIDS = None
OOD_CLASS_NAMES = None

if OOD_STATS_PATH.exists():
    try:
        stats = np.load(str(OOD_STATS_PATH), allow_pickle=True)
        OOD_CENTROIDS = stats["centroids"]  # Shape (35, dim)
        OOD_CLASS_NAMES = list(stats["class_names"])
        print(f"[OOD] Successfully loaded {len(OOD_CLASS_NAMES)} class centroids from {OOD_STATS_PATH.name}")
    except Exception as e:
        print(f"[OOD WARNING] Could not load centroids from {OOD_STATS_PATH}: {e}")
else:
    print(f"[OOD WARNING] {OOD_STATS_PATH} not found. Fallback to confidence threshold ({MIN_CONFIDENCE}).")

# =============================================================================
# 7 Supported Crops Metadata
# =============================================================================
CROPS_META = {
    "Banana": {
        "prefix": "Banana_",
        "aliases": ["banana", "කෙසෙල්"],
        "name_en": "Banana",
        "name_si": "කෙසෙල්",
        "botanical": ["musa"],
    },
    "Chilli": {
        "prefix": "Bell_Pepper_",
        "aliases": ["chilli", "chili", "bell pepper", "pepper", "capsicum", "මිරිස්", "මාළු මිරිස්"],
        "name_en": "Chilli / Bell Pepper",
        "name_si": "මිරිස්",
        "botanical": ["capsicum"],
    },
    "Corn": {
        "prefix": "Corn_",
        "aliases": ["corn", "maize", "ඉරිඟු", "බඩඉරිඟු"],
        "name_en": "Corn / Maize",
        "name_si": "ඉරිඟු",
        "botanical": ["zea mays", "zea"],
    },
    "Potato": {
        "prefix": "Potato_",
        "aliases": ["potato", "අර්තාපල්"],
        "name_en": "Potato",
        "name_si": "අර්තාපල්",
        "botanical": ["solanum tuberosum"],
    },
    "Rice": {
        "prefix": "Rice_",
        "aliases": ["rice", "paddy", "වී", "ගොයම්"],
        "name_en": "Rice",
        "name_si": "වී",
        "botanical": ["oryza sativa", "oryza"],
    },
    "Tea": {
        "prefix": "Tea_",
        "aliases": ["tea", "තේ"],
        "name_en": "Tea",
        "name_si": "තේ",
        "botanical": ["camellia sinensis", "camellia"],
    },
    "Tomato": {
        "prefix": "Tomato_",
        "aliases": ["tomato", "තක්කාලි"],
        "name_en": "Tomato",
        "name_si": "තක්කාලි",
        "botanical": ["solanum lycopersicum", "lycopersicon"],
    },
}

UNSUPPORTED_KEYWORDS = [
    "mango", "mangifera", "papaya", "carica", "pawpaw", "coconut", "cocos",
    "palm", "avocado", "guava", "orange", "citrus", "lemon", "lime",
    "cinnamon", "rubber", "cassava", "manioc", "jackfruit", "breadfruit",
    "coffee", "cocoa", "durian", "rambutan", "apple", "grape", "peach",
    "cucumber", "eggplant", "brinjal", "okra", "onion", "garlic", "carrot",
    "other", "unknown"
]

def find_crop_by_name(crop_str: Optional[str]):
    if not crop_str:
        return None
    s = crop_str.strip().lower()
    for crop_key, meta in CROPS_META.items():
        if any(alias in s for alias in meta["aliases"]):
            return crop_key, meta
    return None

def find_crop_by_class_name(class_name: str):
    for crop_key, meta in CROPS_META.items():
        if class_name.startswith(meta["prefix"]):
            return crop_key, meta
    return None

def find_crop_by_species(species_str: Optional[str]):
    if not species_str:
        return None
    s = species_str.strip().lower()
    for crop_key, meta in CROPS_META.items():
        if any(bot in s for bot in meta["botanical"]) or any(alias in s for alias in meta["aliases"]):
            return crop_key, meta
    return None

# PlantNet API key configuration
PLANTNET_API_KEY = os.environ.get("PLANTNET_API_KEY", "")
if not PLANTNET_API_KEY:
    env_file = BASE_DIR.parent / "server" / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("PLANTNET_API_KEY="):
                    PLANTNET_API_KEY = line.split("=", 1)[1].strip()
                    break
        except Exception:
            pass

def query_plantnet(image_bytes: bytes) -> dict:
    """Queries PlantNet for botanical advisory identification. Returns dict with match and score."""
    if not PLANTNET_API_KEY:
        return {"name": "", "score": 0.0, "full": ""}
    try:
        res = requests.post(
            f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_API_KEY}",
            files={"images": ("leaf.jpg", image_bytes, "image/jpeg")},
            data={"organs": "auto"},
            timeout=3.0,
        )
        if res.status_code == 200:
            data = res.json()
            best = data.get("bestMatch", "")
            results = data.get("results", [])
            score = float(results[0].get("score", 0.0)) if results else 0.0
            common = " ".join(results[0].get("species", {}).get("commonNames", [])) if results else ""
            return {"name": best, "score": score, "full": f"{best} {common}".strip()}
    except Exception as e:
        print("[PlantNet advisory error]", e)
    return {"name": "", "score": 0.0, "full": ""}

def compute_ood_score(x_raw: np.ndarray) -> Optional[float]:
    """Computes max cosine similarity between L2-normalized image embedding and 35 centroids."""
    if feature_model is None or OOD_CENTROIDS is None:
        return None
    emb = feature_model.predict(x_raw, verbose=0)[0]
    norm = np.linalg.norm(emb)
    if norm < 1e-9:
        return 0.0
    emb = emb / norm
    sims = np.dot(OOD_CENTROIDS, emb)
    return float(np.max(sims))

def preprocess(image_bytes: bytes):
    """Loads raw 0-255 RGB image at 224x224. Model normalizes internally."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: Optional[str] = Form(None),
    species: Optional[str] = Form(None),
):
    image_bytes = await file.read()
    x = preprocess(image_bytes)

    # 1. Always compute raw unmasked prediction first
    raw = model.predict(x, verbose=0)[0]
    raw_idx = int(np.argmax(raw))
    raw_class_name = CLASS_NAMES[raw_idx]
    raw_confidence = round(float(raw[raw_idx]), 4)

    # Raw top 3 predictions with true probabilities (no scaling or capping)
    top3_idx = raw.argsort()[-3:][::-1]
    top_3 = [
        {"class_name": CLASS_NAMES[i], "confidence": round(float(raw[i]), 4)}
        for i in top3_idx
    ]

    # Crop detected strictly by the raw model from the 35 classes
    detected_crop_tuple = find_crop_by_class_name(raw_class_name)
    detected_crop = detected_crop_tuple[1]["name_en"] if detected_crop_tuple else None
    detected_crop_si = detected_crop_tuple[1]["name_si"] if detected_crop_tuple else None

    # Crop selected by user in UI dropdown
    selected_crop_tuple = find_crop_by_name(crop)
    selected_crop = selected_crop_tuple[1]["name_en"] if selected_crop_tuple else (crop or None)

    # 2. Embedding similarity (OOD score)
    ood_score = compute_ood_score(x)

    # 3. PlantNet botanical identification (strictly advisory)
    plantnet_info = {"name": "", "score": 0.0, "full": ""}
    if species:
        plantnet_info = {"name": species, "score": 1.0, "full": species}
    else:
        plantnet_info = query_plantnet(image_bytes)

    identified_plant = plantnet_info["full"] or plantnet_info["name"] or None

    # Advisory plant check comparison
    plant_check = "unavailable"
    plantnet_crop_tuple = find_crop_by_species(plantnet_info["full"] or plantnet_info["name"])
    if plantnet_info["name"]:
        if detected_crop_tuple and plantnet_crop_tuple and detected_crop_tuple[0] == plantnet_crop_tuple[0]:
            plant_check = "agree"
        else:
            plant_check = "conflict"

    # 4. Out-of-distribution (OOD) decision logic
    is_supported_plant = True

    if ood_score is not None:
        if ood_score < OOD_THRESHOLD:
            is_supported_plant = False
        elif ood_score < (OOD_THRESHOLD + OOD_BORDERLINE):
            # Borderline case: if PlantNet strongly identifies a species outside the 7 crops
            if plantnet_info["score"] >= PLANTNET_MIN_SCORE:
                matched_supported = find_crop_by_species(plantnet_info["full"] or plantnet_info["name"])
                if not matched_supported:
                    is_supported_plant = False

    # Also treat as unsupported if raw confidence is very low
    if raw_confidence < MIN_CONFIDENCE:
        is_supported_plant = False

    # 5. Determine response status and fields
    if not is_supported_plant:
        # Case "unsupported_crop": return class_name=null, confidence=null, no diagnosis
        status = "unsupported_crop"
        class_name = None
        confidence = None
        top_3_out = []
        crop_mismatch = False
        detected_crop_out = None
        detected_crop_si_out = None
        message_en = (
            "This leaf does not appear to belong to any of the 7 supported crops "
            "(Banana, Chilli / Bell Pepper, Corn / Maize, Potato, Rice, Tea, Tomato). "
            "Please upload a leaf from one of these crops."
        )
        message_si = (
            "මෙම පත්‍රය සහාය දක්වන ප්‍රධාන බෝග 7 න් (කෙසෙල්, මිරිස්, බඩඉරිඟු, අර්තාපල්, වී, තේ, තක්කාලි) "
            "කිසිවකට අයත් නොවන බව පෙනේ. කරුණාකර මෙම බෝගයක පත්‍රයක් උඩුගත කරන්න."
        )
    elif selected_crop_tuple and detected_crop_tuple and selected_crop_tuple[0] != detected_crop_tuple[0]:
        # Case "crop_mismatch": supported but detected != selected.
        # Do NOT mask to selected crop. Return raw top class with TRUE probability.
        status = "crop_mismatch"
        class_name = raw_class_name
        confidence = raw_confidence
        top_3_out = top_3
        crop_mismatch = True
        detected_crop_out = detected_crop
        detected_crop_si_out = detected_crop_si
        message_en = (
            f"You selected {selected_crop}, but the photo looks like {detected_crop}. "
            f"The result below is for {detected_crop}. Do NOT apply treatments meant for {selected_crop}."
        )
        message_si = (
            f"ඔබ {selected_crop} තෝරා ඇතත්, ඡායාරූපය {detected_crop_si} ලෙස පෙනේ. "
            f"පහත ප්‍රතිඵලය {detected_crop_si} සඳහා වේ. {selected_crop} සඳහා වන ප්‍රතිකාර නොයොදන්න."
        )
    else:
        # Case "ok": supported and detected == selected (or no crop selected).
        status = "ok"
        crop_mismatch = False
        detected_crop_out = detected_crop
        detected_crop_si_out = detected_crop_si
        message_en = None
        message_si = None

        if selected_crop_tuple:
            # Restrict and renormalize to selected crop's classes
            prefix = selected_crop_tuple[1]["prefix"]
            mask = np.array([n.startswith(prefix) for n in CLASS_NAMES], dtype=np.float32)
            masked_probs = raw * mask
            sum_p = float(np.sum(masked_probs))
            if sum_p > 0:
                top_idx = int(np.argmax(masked_probs))
                class_name = CLASS_NAMES[top_idx]
                confidence = round(float(masked_probs[top_idx] / sum_p), 4)
                crop_indices = [i for i, n in enumerate(CLASS_NAMES) if n.startswith(prefix)]
                crop_indices.sort(key=lambda i: raw[i], reverse=True)
                top_3_out = [
                    {"class_name": CLASS_NAMES[i], "confidence": round(float(raw[i] / sum_p), 4)}
                    for i in crop_indices[:3]
                ]
            else:
                class_name = raw_class_name
                confidence = raw_confidence
                top_3_out = top_3
        else:
            class_name = raw_class_name
            confidence = raw_confidence
            top_3_out = top_3

    return {
        "status": status,
        "class_name": class_name,
        "confidence": confidence,
        "top_3": top_3_out,
        "raw_class_name": raw_class_name,
        "raw_confidence": raw_confidence,
        "is_supported_plant": is_supported_plant,
        "crop_mismatch": crop_mismatch,
        "selected_crop": selected_crop,
        "detected_crop": detected_crop_out,
        "detected_crop_si": detected_crop_si_out,
        "identified_plant": identified_plant,
        "ood_score": round(ood_score, 4) if ood_score is not None else None,
        "plant_check": plant_check,
        "message_en": message_en,
        "message_si": message_si,
    }
