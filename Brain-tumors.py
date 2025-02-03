import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import matplotlib.pyplot as plt

# Function to display images during processing
def display_images(title, images, cols=2):
    rows = len(images) // cols + (len(images) % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, image) in enumerate(images.items()):
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Function to save images in separate folders
def save_images(image_dict, base_folder="Processed_Images"):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    for name, img in image_dict.items():
        folder_path = os.path.join(base_folder, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = f"{name}.jpg"
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Save with high quality
        print(f"Saved {file_name} in {folder_path}")

# Main Processing Function
def process_image(image_path):
    # Read the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        print("Image not found. Please check the path.")
        return

    processed_images = {'Original': original_image}

    # 1. Contrast Enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(original_image)
    processed_images['Contrast Enhanced'] = enhanced_image

    # 2. Noise Reduction using Gaussian Blur
    denoised_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
    processed_images['Denoised'] = denoised_image

    # 3. Edge Detection using Canny
    edges = cv2.Canny(denoised_image, threshold1=30, threshold2=70)
    processed_images['Edges'] = edges

    # 4. Increase Resolution using Bicubic Interpolation
    resized_image = cv2.resize(denoised_image, (original_image.shape[1]*2, original_image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    processed_images['Resized'] = resized_image

    # 5. Convert to RGB (Optional - for visualization purposes)
    rgb_image = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2RGB)
    processed_images['RGB'] = rgb_image

    # Display the results
    display_images("Processing Steps", processed_images)

    # Save the processed images in separate folders
    save_images(processed_images)

    print("Image processing completed and all enhanced images are saved.")

# Run the script
if __name__ == "__main__":
    # Specify the path to your image
    image_path = 'images\img2.jpg'  # Replace with your image path
    process_image(image_path)