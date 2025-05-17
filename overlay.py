import os
import cv2
import numpy as np

def generate_composite(input_dir, output_path):
    images = []
    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(input_dir, file))
            if img is not None:
                images.append(img)

    if not images:
        print(f"No images found in {input_dir} to composite.")
        return

    stack = np.stack(images, axis=0)
    composite = np.median(stack, axis=0).astype(np.uint8)
    cv2.imwrite(output_path, composite)
    print(f"Composite saved to {output_path}")
