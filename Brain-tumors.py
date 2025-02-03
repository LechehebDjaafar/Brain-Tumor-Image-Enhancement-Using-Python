import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import matplotlib.pyplot as plt
import pywt  # For Wavelet Transform

# Function to display images during processing
def display_images(title, images, cols=2):
    rows = len(images) // cols + (len(images) % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, image) in enumerate(images.items()):
        if len(image.shape) == 2:  # Grayscale
            axes[i].imshow(image, cmap='gray')
        else:  # RGB
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
        
        # Save image with high quality
        if len(img.shape) == 2:  # Grayscale
            cv2.imwrite(file_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:  # RGB
            cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

        print(f"Saved {file_name} in {folder_path}")

# Function to apply Wavelet Transform for noise reduction
def wavelet_denoising(image):
    # Ensure the image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize the image to avoid float64 issues
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Wavelet Transform
    coeffs = pywt.wavedec2(image, 'bior1.3', level=1)
    
    # Separate approximation and detail coefficients
    cA, (cH, cV, cD) = coeffs
    
    # Estimate noise level using MAD
    sigma = mad(cD)  # Use the diagonal detail coefficients
    threshold = sigma * np.sqrt(2 * np.log(image.size))  # Universal threshold
    
    # Denoise using soft thresholding
    cH = pywt.threshold(cH, threshold, mode='soft')
    cV = pywt.threshold(cV, threshold, mode='soft')
    cD = pywt.threshold(cD, threshold, mode='soft')
    
    # Reconstruct the coefficients tuple
    denoised_coeffs = (cA, (cH, cV, cD))
    
    # Reconstruct the image
    denoised_image = pywt.waverec2(denoised_coeffs, 'bior1.3')
    
    # Normalize the result to uint8
    denoised_image = cv2.normalize(denoised_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return denoised_image

# Median Absolute Deviation function
def mad(data):
    if len(data) == 0:
        return 0
    median = np.median(data)
    return np.median(np.abs(data - median))

# Main Processing Function
def process_image(image_path, resize_factor=2):
    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Image not found. Please check the path.")
        return

    # Convert to grayscale if needed
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    processed_images = {'Original': original_image}

    # 1. Contrast Enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_image)
    processed_images['Contrast_Enhanced'] = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    # 2. Noise Reduction using Gaussian Blur
    denoised_gray = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)
    processed_images['Denoised'] = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)

    # 3. Advanced Noise Reduction using Wavelet Transform
    try:
        wavelet_denoised = wavelet_denoising(denoised_gray)
        processed_images['Wavelet_Denoised'] = cv2.cvtColor(wavelet_denoised, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"Error during Wavelet Denoising: {e}")
        processed_images['Wavelet_Denoised'] = cv2.cvtColor(denoised_gray, cv2.COLOR_GRAY2BGR)

    # 4. Edge Detection using Canny
    # Normalize the image before applying Canny
    edges = cv2.Canny(wavelet_denoised, threshold1=30, threshold2=70)
    processed_images['Edges'] = edges

    # 5. Increase Resolution using Bicubic Interpolation
    resized_image = cv2.resize(wavelet_denoised, (wavelet_denoised.shape[1]*resize_factor, wavelet_denoised.shape[0]*resize_factor), interpolation=cv2.INTER_CUBIC)
    processed_images['Resized'] = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)

    # 6. Create Composite Image (Original + Enhanced)
    composite = np.hstack((cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), processed_images['Contrast_Enhanced']))
    processed_images['Composite'] = composite

    # Display the results
    display_images("Processing Steps", processed_images)

    # Save the processed images in separate folders
    save_images(processed_images)

    print("Image processing completed and all enhanced images are saved.")

# Run the script
if __name__ == "__main__":
    # Specify the path to your image
    image_path = 'images/img2.jpg'  # Replace with your image path
    process_image(image_path, resize_factor=2)