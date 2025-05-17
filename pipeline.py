import os
import shutil
from face import process_faces
from overlay import generate_composite

# === CONFIGURATION ===
TITLE_FILE = "title.txt"
IMAGE_ROOTS = ["SD2.1"]
OUTPUT_ROOT = "output"

def load_titles(title_file):
    with open(title_file, "r") as f:
        return [line.strip().lower() for line in f if line.strip()]

def match_title_in_filename(filename, titles):
    name_lower = filename.lower()
    for title in titles:
        if title in name_lower:
            return title
    return None

def organize_images_by_title(source_folder, titles, base_output_dir):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            title_match = match_title_in_filename(file, titles)
            if title_match:
                title_folder = os.path.join(base_output_dir, title_match)
                os.makedirs(title_folder, exist_ok=True)
                shutil.copy(os.path.join(root, file), os.path.join(title_folder, file))

def process_root_folder(source_root, titles):
    output_path = os.path.join(OUTPUT_ROOT, source_root)
    os.makedirs(output_path, exist_ok=True)

    # Step 1: Organize images into title folders
    organize_images_by_title(source_root, titles, output_path)

    composite_collection_dir = os.path.join(output_path, "all_composites")
    os.makedirs(composite_collection_dir, exist_ok=True)

    # Step 2–3: For each title folder, filter faces + generate composite
    for title in titles:
        folder_path = os.path.join(output_path, title)
        if not os.path.isdir(folder_path):
            continue

        filtered_dir = os.path.join(folder_path, "filtered")
        process_faces(folder_path, filtered_dir)

        composite_path = os.path.join(folder_path, "composite.png")
        generate_composite(filtered_dir, composite_path)

        # Step 4: Copy composite to collection folder
        if os.path.exists(composite_path):
            shutil.copy(composite_path, os.path.join(composite_collection_dir, f"{title}_composite.png"))

def main():
    titles = load_titles(TITLE_FILE)
    for root in IMAGE_ROOTS:
        process_root_folder(root, titles)

if __name__ == "__main__":
    main()
