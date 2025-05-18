from datasets import load_dataset
import requests
import os
import re

# Load a small chunk of LAION-Aesthetic (subset of LAION ranking 6+ on the aesthetic score)
dataset = load_dataset("laion/laion2B-en-aesthetic", split="train", streaming=True)
word_pattern = re.compile(r"\bYOURWORDHERE\b", re.IGNORECASE) # replace YOURWORDHERE with the word you want to search for
banned_keywords = ["logo", "sign", "label", "icon", "symbol", "badge", "emblem", "book", "cover", "mug", "shirt", "design"] #filter out images that contain these words

# Filter captions that mention "nurse"
filtered = (
    item for item in dataset
    if (caption := (item.get("TEXT") or "").lower())
    and word_pattern.search(caption)
    and not any (bad_word in caption for bad_word in banned_keywords)
    and not any(bad in caption for bad in banned_keywords)
    and item.get("pwatermark") is not None and item["pwatermark"] < 0.5 #filter out images with watermark
    and item.get("similarity") is not None and item["similarity"] > 0.4 #filter out images with low similarity
    and item.get("punsafe") is not None and item["punsafe"] < 0.4 #filter out images with unsafe content
)

# Directory to save images
os.makedirs("LAION_images", exist_ok=True)

# Download a few examples
for i, item in enumerate(filtered):
    if i >= 250:  # limit for testing
        break
    url = item["URL"]
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(f"LAION_images/image_{i}.jpg", "wb") as f:
                f.write(response.content)
            print(f"Downloaded image_{i}.jpg")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
