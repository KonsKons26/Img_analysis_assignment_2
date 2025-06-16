import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from src.utils import get_paths


def merge_splits(
        base_dir: str,
        save_dir: str
    ) -> None:
    """
    Merges the training and testing datasets into a single directory structure.

    The merged directory will contain two subdirectories: 'NORMAL' and
    'PNEUMONIA', each containing the respective images from both training and
    testing datasets.

    Parameters
    ----------
    base_dir : str
        The base directory containing 'train' and 'test' directories.
    save_dir : str
        The directory where the merged dataset will be saved.

    Returns
    -------
    issues : list
        A list of paths that encountered issues during the merging process.
        If no issues occurred, the list will be empty.
    """

    paths = get_paths(base_dir)

    normals = [
        v["NORMAL"] for v in paths.values()
    ]
    pneumonias = [
        v["PNEUMONIA"] for v in paths.values()
    ]

    os.makedirs(save_dir, exist_ok=True)

    normal_dir = os.path.join(save_dir, "NORMAL")
    pneumonia_dir = os.path.join(save_dir, "PNEUMONIA")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)

    issues = []

    for normals_source_dir in normals:
        for normal in os.listdir(normals_source_dir):
            current_normal_path = os.path.join(
                normals_source_dir, normal
            )
            try:
                shutil.copy(current_normal_path, normal_dir)
            except Exception as e:
                print(f"Error: {e} for {current_normal_path}")
                issues.append(current_normal_path)

    for pneumonia_source_dir in pneumonias:
        for pneumonia in os.listdir(pneumonia_source_dir):
            current_pneumonia_path = os.path.join(
                pneumonia_source_dir, pneumonia
            )
            try:
                shutil.copy(current_pneumonia_path, pneumonia_dir)
            except Exception as e:
                print(f"Error: {e} for {current_pneumonia_path}")
                issues.append(current_pneumonia_path)

    if issues:
        print("Some issues occurred during merging:")
        for issue in issues:
            print(issue)
        return issues

    print("Merging completed successfully.")
    return []


def balance_dataset(in_dir: str, out_dir: str, target_size: tuple = (224, 224)):
    """
    Balance dataset by augmenting images to match class counts.
    All original images are also augmented.
    
    Args:
        in_dir: Input directory containing NORMAL and PNEUMONIA folders
        out_dir: Output directory where balanced dataset will be saved
        target_size: Target size for all images as (width, height). Default is (224, 224)
    """
    
    # Create output directories
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "NORMAL"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "PNEUMONIA"), exist_ok=True)
    
    # Count the normal and pneumonia images
    normal_count = len(os.listdir(os.path.join(in_dir, "NORMAL")))
    pneumonia_count = len(os.listdir(os.path.join(in_dir, "PNEUMONIA")))
    
    print(f"Original counts - NORMAL: {normal_count}, PNEUMONIA: {pneumonia_count}")
    
    # Determine which class needs augmentation
    max_count = max(normal_count, pneumonia_count)
    normal_needed = max_count - normal_count
    pneumonia_needed = max_count - pneumonia_count
    
    print(f"Augmentations needed - NORMAL: {normal_needed}, PNEUMONIA: {pneumonia_needed}")
    
    def augment_image(image_path, output_path, augmentation_type):
        """Apply various augmentations to an image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return False
                
            height, width = img.shape[:2]
            
            if augmentation_type == 'rotation':
                # Random rotation (-15 to 15 degrees)
                angle = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                img = cv2.warpAffine(img, M, (width, height))
                
            elif augmentation_type == 'horizontal_flip':
                # Horizontal flip
                img = cv2.flip(img, 1)
                
            elif augmentation_type == 'brightness':
                # Brightness adjustment
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Brightness(pil_img)
                factor = random.uniform(0.8, 1.2)
                pil_img = enhancer.enhance(factor)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            elif augmentation_type == 'contrast':
                # Contrast adjustment
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Contrast(pil_img)
                factor = random.uniform(0.8, 1.2)
                pil_img = enhancer.enhance(factor)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
            elif augmentation_type == 'zoom':
                # Random zoom (crop and resize)
                zoom_factor = random.uniform(0.8, 1.2)
                if zoom_factor < 1:
                    # Zoom out (add padding)
                    new_height = int(height / zoom_factor)
                    new_width = int(width / zoom_factor)
                    img = cv2.resize(img, (new_width, new_height))
                    top = (new_height - height) // 2
                    left = (new_width - width) // 2
                    img = img[top:top+height, left:left+width]
                else:
                    # Zoom in (crop)
                    new_height = int(height / zoom_factor)
                    new_width = int(width / zoom_factor)
                    top = (height - new_height) // 2
                    left = (width - new_width) // 2
                    img = img[top:top+new_height, left:left+new_width]
                    img = cv2.resize(img, (width, height))
                    
            elif augmentation_type == 'shift':
                # Random translation
                shift_x = random.randint(-20, 20)
                shift_y = random.randint(-20, 20)
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                img = cv2.warpAffine(img, M, (width, height))
                
            elif augmentation_type == 'noise':
                # Add salt and pepper noise
                noise_fraction = 0.025  # 2.5% of pixels will be affected
                height, width = img.shape[:2]
                total_pixels = height * width
                num_noise_pixels = int(total_pixels * noise_fraction)
                
                # Create a copy to avoid modifying original
                noisy_img = img.copy()
                
                # Add salt noise (white pixels)
                salt_pixels = num_noise_pixels // 2
                for _ in range(salt_pixels):
                    y = random.randint(0, height - 1)
                    x = random.randint(0, width - 1)
                    noisy_img[y, x] = 255  # White pixel
                
                # Add pepper noise (black pixels)
                pepper_pixels = num_noise_pixels - salt_pixels
                for _ in range(pepper_pixels):
                    y = random.randint(0, height - 1)
                    x = random.randint(0, width - 1)
                    noisy_img[y, x] = 0  # Black pixel
                
                img = noisy_img
            
            # Resize to target size regardless of augmentation type
            img = cv2.resize(img, target_size)
                
            # Save augmented image
            cv2.imwrite(output_path, img)
            return True
            
        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return False
    
    def process_class(class_name, needed_augmentations):
        """Process images for a specific class"""
        class_dir = os.path.join(in_dir, class_name)
        out_class_dir = os.path.join(out_dir, class_name)
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        augmentation_types = ['rotation', 'horizontal_flip', 'brightness', 
                            'contrast', 'zoom', 'shift', 'noise']
        
        # Copy or augment all original images
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(class_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            ext = os.path.splitext(img_file)[1]

            copy_rand = random.choice([True, False])
            if copy_rand:
                # Copy original image (but still resize to target size)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, target_size)
                    original_out_path = os.path.join(out_class_dir, f"{base_name}_original{ext}")
                    cv2.imwrite(original_out_path, img)
            else:
                # Create one augmentation of the original image
                aug_type = augmentation_types[i % len(augmentation_types)]
                aug_out_path = os.path.join(out_class_dir, f"{base_name}_aug_{aug_type}{ext}")
                augment_image(img_path, aug_out_path, aug_type)
        
        # Generate additional augmentations for class balancing
        if needed_augmentations > 0:
            print(f"Generating {needed_augmentations} additional augmentations for {class_name}")
            
            for i in range(needed_augmentations):
                # Select random ORIGINAL image to augment (from input directory, not output)
                source_img = random.choice(image_files)
                source_path = os.path.join(class_dir, source_img)  # This is the original input path
                
                # Select random augmentation type
                aug_type = random.choice(augmentation_types)
                
                # Create output filename
                base_name = os.path.splitext(source_img)[0]
                ext = os.path.splitext(source_img)[1]
                aug_out_path = os.path.join(out_class_dir, 
                                          f"{base_name}_balance_{i}_{aug_type}{ext}")
                
                # Apply augmentation to the original source image
                success = augment_image(source_path, aug_out_path, aug_type)
                if not success:
                    print(f"Failed to create augmentation {i} for {class_name}")
    
    # Process both classes
    process_class("NORMAL", normal_needed)
    process_class("PNEUMONIA", pneumonia_needed)
    
    # Final count verification
    final_normal = len(os.listdir(os.path.join(out_dir, "NORMAL")))
    final_pneumonia = len(os.listdir(os.path.join(out_dir, "PNEUMONIA")))
    
    print(f"\nFinal counts - NORMAL: {final_normal}, PNEUMONIA: {final_pneumonia}")
    print(f"Dataset balancing complete! Output saved to: {out_dir}")