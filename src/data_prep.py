import os
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2

def augment_image(image):
    """Apply various augmentations to an image for better training"""
    augmentations = []

    # Original
    augmentations.append(image)

    # Grayscale
    augmentations.append(ImageOps.grayscale(image).convert('RGB'))

    # Black and White
    gray = ImageOps.grayscale(image)
    augmentations.append(gray.point(lambda x: 0 if x < 128 else 255, '1').convert('RGB'))

    # Histogram equalization
    img_array = np.array(image)
    img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    augmentations.append(Image.fromarray(img_output))

    # Gaussian blur
    augmentations.append(image.filter(ImageFilter.GaussianBlur(radius=1)))

    # Edge enhancement
    augmentations.append(image.filter(ImageFilter.EDGE_ENHANCE))

    # Random rotation (slight)
    augmentations.append(image.rotate(15, expand=True))
    augmentations.append(image.rotate(-15, expand=True))

    # Flip horizontal
    augmentations.append(ImageOps.mirror(image))

    return augmentations

def load_images_from_folder(folder, label, img_size=(224, 224), max_images=None, augment=True):
    images = []
    labels = []
    filenames = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if max_images:
        filenames = filenames[:max_images]  # Limit to max_images
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)

            if augment:
                # Apply augmentations
                augmented_images = augment_image(img)
                for aug_img in augmented_images:
                    img_array = np.array(aug_img) / 255.0  # Normalize to [0,1]
                    images.append(img_array)
                    labels.append(label)
            else:
                img_array = np.array(img) / 255.0  # Normalize to [0,1]
                images.append(img_array)
                labels.append(label)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return images, labels

def prepare_data():
    # Load data with balanced sampling (10 images per class for training, no augmentation for validation)
    corn_images, corn_labels = load_images_from_folder('data/corn', 0, max_images=10, augment=False)  # 0 for corn
    grape_images, grape_labels = load_images_from_folder('data/grape', 1, max_images=10, augment=False)  # 1 for grape
    non_leaf_images, non_leaf_labels = load_images_from_folder('data/non_leaf', 2, max_images=10, augment=False)  # 2 for non-leaf

    # Combine for leaf detector (leaf vs non-leaf)
    leaf_images = corn_images + grape_images
    leaf_labels = [1] * len(leaf_images)  # 1 for leaf
    non_leaf_labels_binary = [0] * len(non_leaf_images)  # 0 for non-leaf

    leaf_detector_images = leaf_images + non_leaf_images
    leaf_detector_labels = leaf_labels + non_leaf_labels_binary

    # Species data (corn and grape only)
    species_images = corn_images + grape_images
    species_labels = corn_labels + grape_labels

    # Convert to numpy arrays (ensure all images are same shape)
    leaf_detector_images = np.array(leaf_detector_images)
    leaf_detector_labels = np.array(leaf_detector_labels)
    species_images = np.array(species_images)
    species_labels = np.array(species_labels)

    # Split data
    # Leaf detector
    X_train_leaf, X_temp_leaf, y_train_leaf, y_temp_leaf = train_test_split(
        leaf_detector_images, leaf_detector_labels, test_size=0.3, random_state=42
    )
    X_val_leaf, X_test_leaf, y_val_leaf, y_test_leaf = train_test_split(
        X_temp_leaf, y_temp_leaf, test_size=0.33, random_state=42
    )

    # Species classifier
    X_train_species, X_temp_species, y_train_species, y_temp_species = train_test_split(
        species_images, species_labels, test_size=0.3, random_state=42
    )
    X_val_species, X_test_species, y_val_species, y_test_species = train_test_split(
        X_temp_species, y_temp_species, test_size=0.33, random_state=42
    )

    return {
        'leaf_detector': {
            'train': (X_train_leaf, y_train_leaf),
            'val': (X_val_leaf, y_val_leaf),
            'test': (X_test_leaf, y_test_leaf)
        },
        'species_classifier': {
            'train': (X_train_species, y_train_species),
            'val': (X_val_species, y_val_species),
            'test': (X_test_species, y_test_species)
        }
    }

if __name__ == "__main__":
    data = prepare_data()
    print("Data prepared successfully!")
    print(f"Leaf detector train: {data['leaf_detector']['train'][0].shape}")
    print(f"Species classifier train: {data['species_classifier']['train'][0].shape}")
