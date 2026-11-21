import os, sys, random, requests

DEFAULT_DIR = r"C:\Users\anush\OneDrive\Desktop\Projects for money\AgriSL\AgriSL_Dataset\train"
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATASET_PATH", DEFAULT_DIR)

if not os.path.exists(DATA_DIR):
    print(f"Error: Dataset directory not found at: {DATA_DIR}")
    print("Usage: python test_accuracy.py <path_to_train_dataset>")
    print("Or set the environment variable DATASET_PATH.")
    sys.exit(1)
URL = "http://127.0.0.1:8001/predict"
PER_CLASS = 10
random.seed(1)

total = correct = 0
first = True
for cls in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, cls)
    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample = random.sample(files, min(PER_CLASS, len(files)))
    ok, wrong = 0, {}
    for f in sample:
        with open(os.path.join(folder, f), "rb") as fh:
            r = requests.post(URL, files={"file": (f, fh, "image/jpeg")}, timeout=120)
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text[:500])
            sys.exit(1)
        data = r.json()
        if first:
            print("Sample response:", data)
            first = False
        pred = data.get("class_name")
        if pred == cls:
            ok += 1
        else:
            wrong[pred] = wrong.get(pred, 0) + 1
    total += len(sample)
    correct += ok
    print(f"{cls:35s} {ok}/{len(sample)}  {wrong if wrong else ''}")

print(f"\nOVERALL: {correct}/{total} = {100*correct/total:.1f}%")
