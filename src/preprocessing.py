import os
import shutil
import random
from PIL import Image
import torchvision.transforms as transforms
from src.utils import get_paths
from typing import Dict, List, Optional


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


def get_augmentation_transforms(intensity='moderate', img_size=224):
    """
    Get augmentation transforms that will be applied and saved as new images.

    Parameters
    ----------
    intensity : str
        Intensity of augmentation: 'light', 'moderate', or 'strong'.
    img_size : int
        Size to which images will be resized before augmentation.

    Returns
    -------
    augmentation_variants : list
        List of augmentation transforms to be applied to images.
    """
    base_transforms = [
        transforms.Resize((img_size, img_size)),
    ]
    
    # Define different augmentation variants
    if intensity == 'light':
        augmentation_variants = [
            transforms.Compose(base_transforms + [
                transforms.RandomRotation(degrees=5, fill=0),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]),
            # Always flip for this variant
            transforms.Compose(base_transforms + [
                transforms.RandomHorizontalFlip(p=1.0),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]),
        ]
    elif intensity == 'moderate':
        augmentation_variants = [
            transforms.Compose(base_transforms + [
                transforms.RandomRotation(degrees=10, fill=0),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomRotation(degrees=7, fill=0),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomPerspective(distortion_scale=0.1, p=1.0),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]),
        ]
    elif intensity == 'strong':
        augmentation_variants = [
            transforms.Compose(base_transforms + [
                transforms.RandomRotation(degrees=15, fill=0),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(degrees=5, fill=0),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            ]),
            transforms.Compose(base_transforms + [
                transforms.RandomPerspective(distortion_scale=0.2, p=1.0),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
            ]),
            transforms.Compose(base_transforms + [
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ]),
        ]
    else:
        augmentation_variants = []

    return augmentation_variants


def apply_and_save_augmentations(
    image_path: str, 
    output_dir: str, 
    base_filename: str,
    augmentation_transforms: List[transforms.Compose],
    num_augmentations_per_variant: int = 1
) -> List[str]:
    """
    Apply augmentations to an image and save the results.

    Parameters
    ----------
    image_path : str
        Path to the original image file.
    output_dir : str
        Directory where augmented images will be saved.
    base_filename : str
        Base filename for the augmented images (without extension).
    augmentation_transforms : List[transforms.Compose]
        List of augmentation transforms to apply.
    num_augmentations_per_variant : int, optional
        Number of augmented images to generate per variant (default is 1).

    Returns
    -------
    List[str]
        List of paths to the saved augmented images.
    """
    try:
        # Load original image
        original_image = Image.open(image_path).convert('RGB')
        saved_paths = []

        # Detect original format for saving
        original_format = original_image.format if original_image.format else 'JPEG'

        # Apply each augmentation variant
        for variant_idx, transform in enumerate(augmentation_transforms):
            for aug_idx in range(num_augmentations_per_variant):
                # Apply transform
                augmented_image = transform(original_image)

                # Convert back to PIL if it's a tensor
                if hasattr(augmented_image, 'numpy'):
                    # Convert tensor back to PIL Image
                    augmented_image = transforms.ToPILImage()(augmented_image)

                # Generate filename with proper extension
                name, ext = os.path.splitext(base_filename)
                if not ext:  # No extension found
                    ext = '.jpeg'  # Default to .jpeg for X-ray images
                aug_filename = f"{name}_aug_v{variant_idx}_n{aug_idx}{ext}"
                aug_path = os.path.join(output_dir, aug_filename)

                # Save augmented image with format specification
                if original_format in ['JPEG', 'JPG'] or ext.lower() in ['.jpg', '.jpeg']:
                    augmented_image.save(aug_path, 'JPEG', quality=95)
                elif original_format == 'PNG' or ext.lower() == '.png':
                    augmented_image.save(aug_path, 'PNG')
                else:
                    # Default to JPEG for unknown formats
                    augmented_image.save(aug_path, 'JPEG', quality=95)

                saved_paths.append(aug_path)

        return saved_paths

    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")
        return []


def split_dataset_with_augmentation(
    inpath: str,
    savepath: str,
    train_frac: float,
    test_frac: float,
    val_frac: float,
    augment_config: Optional[Dict] = None,
    seed: int = None
) -> None:
    """
    Splits the dataset into train, test, and validation sets with optional augmentation.
    All images are resized to the target size regardless of augmentation.
    
    Parameters
    ----------
    inpath : str
        Path to the directory containing 'NORMAL' and 'PNEUMONIA' folders
    savepath : str
        Path where the 'train', 'test', and 'val' directories will be created
    train_frac : float
        Fraction of images to use for training (0-1)
    test_frac : float
        Fraction of images to use for testing (0-1)
    val_frac : float
        Fraction of images to use for validation (0-1)
    augment_config : dict, optional
        Configuration for augmentation and resizing. Example:
        {
            'apply_to_splits': ['train'],  # Which splits to augment (all splits get resized)
            'intensity': 'moderate',       # 'light', 'moderate', 'strong'
            'augmentations_per_image': 3,  # How many augmented versions per original
            'variants_per_image': 2,       # How many variants to use per image
            'balance_classes': True,       # Balance classes using augmentation
            'img_size': 224               # Target image size (applied to ALL splits)
        }
    seed : int, optional
        Random seed for reproducibility
    """

    # Validate fractions sum to 1
    if not (0.9999 <= train_frac + test_frac + val_frac <= 1.0001):
        raise ValueError("Fractions must sum to 1")

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Default augmentation config
    if augment_config is None:
        augment_config = {
            'apply_to_splits': [],
            'intensity': 'moderate',
            'augmentations_per_image': 2,
            'variants_per_image': 2,
            'balance_classes': False,
            'img_size': 224
        }

    # Create output directory structure
    splits = ['train', 'test', 'val']
    classes = ['NORMAL', 'PNEUMONIA']

    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(savepath, split, class_name), exist_ok=True)

    # Get augmentation transforms if needed
    augmentation_transforms = []
    if augment_config['apply_to_splits']:
        all_transforms = get_augmentation_transforms(
            intensity=augment_config['intensity'],
            img_size=augment_config['img_size']
        )
        # Limit to specified number of variants
        augmentation_transforms = all_transforms[:augment_config['variants_per_image']]

    # Create base resize transform for all images
    base_resize_transform = transforms.Compose([
        transforms.Resize((augment_config['img_size'], augment_config['img_size']))
    ])

    # Count images per class for balancing
    class_counts = {}
    for class_name in classes:
        class_path = os.path.join(inpath, class_name)
        all_files = os.listdir(class_path)
        images = []

        for f in all_files:
            # Check if it's an image file by extension or by trying to open it
            if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                images.append(f)
            else:
                # Try to open as image to verify it's an image file
                try:
                    test_path = os.path.join(class_path, f)
                    with Image.open(test_path) as img:
                        images.append(f)  # It's a valid image
                except:
                    continue  # Skip non-image files

        class_counts[class_name] = len(images)

    # Determine if we need to balance and which class needs more augmentation
    balance_info = {}
    if augment_config.get('balance_classes', False):
        max_count = max(class_counts.values())
        for class_name, count in class_counts.items():
            if count < max_count:
                balance_info[class_name] = max_count - count

    # Process each class
    total_original = 0
    total_augmented = 0

    for class_name in classes:
        print(f"Processing {class_name} class...")

        # Get all image paths for this class
        class_path = os.path.join(inpath, class_name)
        # Include files without extensions (common in medical datasets)
        all_files = os.listdir(class_path)
        images = []

        for f in all_files:
            # Check if it's an image file by extension or by trying to open it
            if f.lower().endswith(('.jpeg', '.jpg', '.png')):
                images.append(f)
            else:
                # Try to open as image to verify it's an image file
                try:
                    test_path = os.path.join(class_path, f)
                    with Image.open(test_path) as img:
                        images.append(f)  # It's a valid image
                except:
                    continue  # Skip non-image files

        # Shuffle the images
        random.shuffle(images)

        # Calculate split indices
        num_images = len(images)
        train_end = int(train_frac * num_images)
        test_end = train_end + int(test_frac * num_images)

        # Split into train, test, val
        splits_images = {
            'train': images[:train_end],
            'test': images[train_end:test_end],
            'val': images[test_end:]
        }

        # Copy files and apply augmentation where specified
        for split_name, split_images in splits_images.items():
            print(f"  Processing {split_name} split: {len(split_images)} images")

            split_original_count = 0
            split_augmented_count = 0
            
            for img in split_images:
                src = os.path.join(class_path, img)
                dst_dir = os.path.join(savepath, split_name, class_name)
                dst = os.path.join(dst_dir, img)

                # Handle potential file existence
                if os.path.exists(dst):
                    base, ext = os.path.splitext(img)
                    counter = 1
                    while True:
                        new_name = f"{base}_{counter}{ext}"
                        new_dst = os.path.join(dst_dir, new_name)
                        if not os.path.exists(new_dst):
                            dst = new_dst
                            img = new_name  # Update for augmentation naming
                            break
                        counter += 1

                # Load and resize original image (all splits get resized)
                try:
                    original_image = Image.open(src).convert('RGB')
                    resized_image = base_resize_transform(original_image)

                    # Save resized original image
                    resized_image.save(dst)
                    split_original_count += 1

                    # Apply augmentation if specified for this split
                    if (split_name in augment_config['apply_to_splits'] and 
                        augmentation_transforms):

                        base_filename = os.path.splitext(img)[0]

                        # Standard augmentation (using the already resized image)
                        augmented_paths = apply_and_save_augmentations(
                            dst, dst_dir, img, augmentation_transforms,
                            num_augmentations_per_variant=1
                        )
                        split_augmented_count += len(augmented_paths)

                        # Additional augmentation for class balancing
                        if class_name in balance_info and split_name == 'train':
                            extra_augs_needed = balance_info[class_name] // len(splits_images[split_name])
                            if extra_augs_needed > 0:
                                extra_augmented_paths = apply_and_save_augmentations(
                                    dst, dst_dir, f"{base_filename}_balance", 
                                    augmentation_transforms,
                                    num_augmentations_per_variant=extra_augs_needed
                                )
                                split_augmented_count += len(extra_augmented_paths)

                except Exception as e:
                    print(f"Error processing {src}: {e}")
                    # Fallback to simple copy if resize fails
                    shutil.copy2(src, dst)
                    split_original_count += 1

            print(f"    Original: {split_original_count}, Augmented: {split_augmented_count}")
            total_original += split_original_count
            total_augmented += split_augmented_count

    print("\nDataset splitting with augmentation completed successfully.")
    print(f"Directory structure created at: {savepath}")
    print(f"Total original images: {total_original}")
    print(f"Total augmented images: {total_augmented}")
    print(f"Total images: {total_original + total_augmented}")
