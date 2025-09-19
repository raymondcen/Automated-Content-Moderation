import os
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
from PIL import Image
import shutil

# Paths
nsfw_img_path = Path("c:/Users/raymo/Desktop/nsfw_img_resized")
mmhs150k_img_dataset_path = Path("c:/Users/raymo/Desktop/img_resized")

nsfw_img_path.mkdir(exist_ok=True)

# Load classifier
classifier = pipeline(
    "image-classification",
    model="Falconsai/nsfw_image_detection",
    device=0,
    dtype="auto"
)

# Collect image paths
paths = list(mmhs150k_img_dataset_path.glob("*.jpg"))
paths = paths[::-1]  # optional: start from the end

batch_size = 48
prefetch_batches = 3  # load 3 batches at a time to limit RAM usage
flagged = []

progress = tqdm(total=len(paths), desc="Processing", unit="img")

# Process images in prefetch batches
for i in range(0, len(paths), batch_size * prefetch_batches):
    batch_paths = paths[i:i + batch_size * prefetch_batches]
    
    # Preload images in this prefetch
    images = [Image.open(p).convert("RGB") for p in batch_paths]

    # Process each smaller batch
    for j in range(0, len(images), batch_size):
        sub_images = images[j:j+batch_size]
        sub_paths = batch_paths[j:j+batch_size]
        
        results = classifier(sub_images, batch_size=len(sub_images))

        for path, preds in zip(sub_paths, results):
            top = max(preds, key=lambda x: x["score"])
            if top["label"].lower() == "nsfw":
                flagged.append(path)
        
        progress.update(len(sub_images))

progress.close()

# Move flagged files
print(f"\nMoving {len(flagged)} NSFW images to {nsfw_img_path}...")
for path in tqdm(flagged, desc="Moving", unit="img"):
    dest = nsfw_img_path / path.name
    try:
        path.rename(dest)
    except Exception:
        shutil.move(str(path), str(dest))

print("Sift completed")
