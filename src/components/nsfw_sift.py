import os
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm 

# Paths
img_dir = Path("c:/Users/raymo/Desktop/img_resized")
quarantine_dir = Path("c:/Users/raymo/Desktop/quarantine")
quarantine_dir.mkdir(exist_ok=True)

# Load classifier
classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection",
    device=0,
    torch_dtype="auto"
)

# Collect images
paths = list(img_dir.glob("*.jpg"))
paths = paths[::-1]  # start at end

batch_size = 48  # adjust if OOM
progress = tqdm(total=len(paths), desc="Processing", unit="img")

flagged = []  # store NSFW images to move later

# Loop with progress tracking
for i in range(0, len(paths), batch_size):
    batch_paths = paths[i:i+batch_size]

    results = classifier([str(p) for p in batch_paths], batch_size=batch_size)

    for path, preds in zip(batch_paths, results):
        top = max(preds, key=lambda x: x["score"])
        if top["label"].lower() == "nsfw":
            flagged.append(path)

    progress.update(len(batch_paths))  # advance bar

progress.close()

# Move flagged files after classification
print(f"\nMoving {len(flagged)} NSFW images to quarantine...")
for path in tqdm(flagged, desc="Moving", unit="img"):
    dest = quarantine_dir / path.name
    try:
        path.rename(dest)
    except Exception as e:
        print(f"Failed to move {path}: {e}")

print("Sift completed")
