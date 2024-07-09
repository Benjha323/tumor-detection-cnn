import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# Paths to the image directories
image_dirs = {
    "colon_aca": r"C:\Users\Benja\Desktop\pruebas cnn\Lung and Colon Cancer Histopathological Images\lung_colon_image_set\colon_image_sets\colon_aca",
    "colon_n": r"C:\Users\Benja\Desktop\pruebas cnn\Lung and Colon Cancer Histopathological Images\lung_colon_image_set\colon_image_sets\colon_n",
    "lung_aca": r"C:\Users\Benja\Desktop\pruebas cnn\Lung and Colon Cancer Histopathological Images\lung_colon_image_set\lung_image_sets\lung_aca",
    "lung_n": r"C:\Users\Benja\Desktop\pruebas cnn\Lung and Colon Cancer Histopathological Images\lung_colon_image_set\lung_image_sets\lung_n",
    "lung_scc": r"C:\Users\Benja\Desktop\pruebas cnn\Lung and Colon Cancer Histopathological Images\lung_colon_image_set\lung_image_sets\lung_scc",
}

# Output size for images
output_size = (256, 256)

def preprocess_images(image_dirs, output_size):
    for label, dir_path in image_dirs.items():
        output_dir = os.path.join("processed_images", label)
        os.makedirs(output_dir, exist_ok=True)
        images = os.listdir(dir_path)
        
        for img_name in tqdm(images, desc=f"Processing {label}"):
            img_path = os.path.join(dir_path, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.resize(output_size)
                    img = img.convert("RGB")
                    img.save(os.path.join(output_dir, img_name))
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

preprocess_images(image_dirs, output_size)
