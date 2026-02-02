import cv2
import os
import numpy as np
from tqdm import tqdm  # For progress bar

def create_output_directory(base_output_path):
    """Create output directories for different enhancement methods"""
    directories = {
        'normalized': os.path.join(base_output_path, 'normalized'),
        'clahe': os.path.join(base_output_path, 'clahe'),
        'histogram_eq': os.path.join(base_output_path, 'histogram_eq'),
        'gamma': os.path.join(base_output_path, 'gamma')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def enhance_depth_image(image):
    """Apply different enhancement methods to the depth image"""
    # Convert to 8-bit if needed
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    enhanced = {}
    
    # 1. Simple normalization
    enhanced['normalized'] = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced['clahe'] = clahe.apply(image)
    
    # 3. Regular histogram equalization
    enhanced['histogram_eq'] = cv2.equalizeHist(image)
    
    # 4. Gamma correction (adjust gamma value as needed)
    gamma = 1.5
    lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
    enhanced['gamma'] = cv2.LUT(image, lookup_table)
    
    return enhanced

def process_depth_images(input_folder, output_base_folder):
    """Process all depth images in the input folder"""
    # Create output directories
    output_dirs = create_output_directory(output_base_folder)
    
    # Get list of PNG files
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        # Read image
        image_path = os.path.join(input_folder, image_file)
        original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if original is None:
            print(f"Failed to read image: {image_file}")
            continue
            
        # Apply enhancements
        enhanced_images = enhance_depth_image(original)
        
        # Save enhanced images
        for method, enhanced_image in enhanced_images.items():
            output_path = os.path.join(output_dirs[method], f"{os.path.splitext(image_file)[0]}_{method}.png")
            cv2.imwrite(output_path, enhanced_image)

def main():
    # Define input and output folders
    input_folder = "/home/harshr/NV_cahsor/CAHSOR-master/data/BL_2024-09-04_19-09-17_chunk0000/dense_depth_map"  # Replace with your input folder path
    output_folder = "/home/harshr/NV_cahsor/CAHSOR-master/data/BL_2024-09-04_19-09-17_chunk0000/depth_imgs"  # Replace with your output folder path
    
    try:
        # Process images
        process_depth_images(input_folder, output_folder)
        print("Processing completed successfully!")
        
        # Optional: Display sample results
        sample_file = os.listdir(input_folder)[0]
        original = cv2.imread(os.path.join(input_folder, sample_file), cv2.IMREAD_UNCHANGED)
        
        # Display sample results (optional)
        cv2.imshow('Original', original)
        enhanced = enhance_depth_image(original)
        for method, img in enhanced.items():
            cv2.imshow(method, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
