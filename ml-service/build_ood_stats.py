import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"
OUTPUT_PATH = BASE_DIR / "model" / "ood_stats.npz"

DEFAULT_DATA_DIR = r"C:\Users\anush\OneDrive\Desktop\Projects for money\AgriSL\AgriSL_Dataset\train"
MAX_SAMPLES_PER_CLASS = 150
IMG_SIZE = (224, 224)

# OneDrive offline flags: FILE_ATTRIBUTE_OFFLINE (0x1000), FILE_ATTRIBUTE_RECALL_ON_OPEN (0x40000), RECALL_ON_DATA_ACCESS (0x400000)
OFFLINE_MASK = 0x1000 | 0x40000 | 0x400000

def find_global_pooling_layer(model):
    """Finds the GlobalAveragePooling2D layer dynamically without hardcoding."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D) or "GlobalAveragePooling2D" in layer.__class__.__name__:
            return layer
    raise ValueError("No GlobalAveragePooling2D layer found in model.")

def is_pure_local_file(p: Path) -> bool:
    try:
        attrs = p.stat().st_file_attributes
        return not bool(attrs & OFFLINE_MASK)
    except Exception:
        return False

def build_centroids(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset directory not found: {data_dir}")

    print(f"Loading model from {MODEL_PATH}...", flush=True)
    model = tf.keras.models.load_model(str(MODEL_PATH))

    pool_layer = find_global_pooling_layer(model)
    print(f"Using feature layer: {pool_layer.name} ({pool_layer.__class__.__name__})", flush=True)
    feature_model = tf.keras.Model(inputs=model.input, outputs=pool_layer.output)

    @tf.function
    def extract_features(batch):
        return feature_model(batch, training=False)

    # Warmup
    dummy = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    extract_features(dummy)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    random.seed(42)
    centroids = []

    print(f"Building class centroids from {data_path} (up to {MAX_SAMPLES_PER_CLASS} local samples/class)...", flush=True)

    for idx, cls_name in enumerate(class_names):
        cls_dir = data_path / cls_name
        if not cls_dir.exists():
            print(f"WARNING: Class directory not found: {cls_dir}", flush=True)
            continue

        image_files = [
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"] and is_pure_local_file(f)
        ]

        if not image_files:
            print(f"WARNING: No pure local images found in {cls_dir}", flush=True)
            continue

        sampled_files = random.sample(image_files, min(len(image_files), MAX_SAMPLES_PER_CLASS))

        batch_images = []
        for img_path in sampled_files:
            try:
                img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32)  # Raw 0-255 RGB
                batch_images.append(arr)
            except Exception as e:
                print(f"Error reading {img_path}: {e}", flush=True)

        if not batch_images:
            continue

        batch_arr = np.array(batch_images, dtype=np.float32)
        features = extract_features(batch_arr).numpy()

        # L2-normalize each embedding
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        norms[norms == 0] = 1e-12
        norm_features = features / norms

        # Class centroid = normalized mean of normalized embeddings
        mean_feat = np.mean(norm_features, axis=0)
        centroid = mean_feat / (np.linalg.norm(mean_feat) + 1e-12)
        centroids.append(centroid)

        print(f"[{idx+1}/{len(class_names)}] {cls_name:35s}: {len(batch_images)} local images -> centroid computed", flush=True)

    centroids_arr = np.array(centroids, dtype=np.float32)
    print(f"\nFinal centroids matrix shape: {centroids_arr.shape}", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(OUTPUT_PATH),
        centroids=centroids_arr,
        class_names=np.array(class_names),
    )
    print(f"Saved OOD centroids to {OUTPUT_PATH}", flush=True)

if __name__ == "__main__":
    target_data_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATA_DIR", DEFAULT_DATA_DIR)
    build_centroids(target_data_dir)
