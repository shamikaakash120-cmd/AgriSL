import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "disease_model.keras"
OOD_STATS_PATH = BASE_DIR / "model" / "ood_stats.npz"
CALIB_DIR = BASE_DIR / "calibration"
SUPPORTED_DIR = CALIB_DIR / "supported"
UNSUPPORTED_DIR = CALIB_DIR / "unsupported"

IMG_SIZE = (224, 224)

def calibrate():
    if not OOD_STATS_PATH.exists():
        print(f"ERROR: {OOD_STATS_PATH} not found. Run build_ood_stats.py first.")
        sys.exit(1)

    stats = np.load(str(OOD_STATS_PATH))
    centroids = stats["centroids"]
    class_names = list(stats["class_names"])
    print(f"Loaded {len(centroids)} class centroids.")

    model = tf.keras.models.load_model(str(MODEL_PATH))
    pool_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D) or "GlobalAveragePooling2D" in layer.__class__.__name__:
            pool_layer = layer
            break
    feature_model = tf.keras.Model(inputs=model.input, outputs=pool_layer.output)

    @tf.function
    def extract_feat(x):
        return feature_model(x, training=False)

    def evaluate_dir(d: Path):
        scores = []
        if not d.exists():
            return scores
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        for f in files:
            try:
                img = Image.open(f).convert("RGB").resize(IMG_SIZE)
                arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
                feat = extract_feat(arr).numpy()
                norm_feat = feat / (np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-12)
                sims = np.dot(norm_feat, centroids.T)[0]
                score = float(np.max(sims))
                pred_cls = class_names[int(np.argmax(sims))]
                raw_pred = model(arr, training=False).numpy()[0]
                raw_conf = float(np.max(raw_pred))
                scores.append({"file": f.name, "ood_score": score, "closest": pred_cls, "raw_conf": raw_conf})
                print(f"  {f.name:30s} -> ood_score: {score:.4f}, raw_conf: {raw_conf:.4f}, closest: {pred_cls}")
            except Exception as e:
                print(f"  Error reading {f.name}: {e}")
        return scores

    print(f"\n--- Evaluating Supported Calibration Set ({SUPPORTED_DIR}) ---")
    sup_scores = evaluate_dir(SUPPORTED_DIR)
    if not sup_scores:
        print("  [Note: supported folder is currently empty]")

    print(f"\n--- Evaluating Unsupported Calibration Set ({UNSUPPORTED_DIR}) ---")
    unsup_scores = evaluate_dir(UNSUPPORTED_DIR)
    if not unsup_scores:
        print("  [Note: unsupported folder is currently empty]")

    if sup_scores and unsup_scores:
        s_vals = [x["ood_score"] for x in sup_scores]
        u_vals = [x["ood_score"] for x in unsup_scores]
        print(f"\nSummary:")
        print(f"  Supported: min={min(s_vals):.4f}, mean={np.mean(s_vals):.4f}, max={max(s_vals):.4f}")
        print(f"  Unsupported: min={min(u_vals):.4f}, mean={np.mean(u_vals):.4f}, max={max(u_vals):.4f}")

if __name__ == "__main__":
    calibrate()
