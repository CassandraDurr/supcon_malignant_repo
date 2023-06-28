"""Supervised contrastive learning using two CNN encoder types image data, with balanced batching."""
import tensorflow as tf
from kerasgen.balanced_image_dataset import balanced_image_dataset_from_directory
from sklearn.model_selection import train_test_split

from functions import (
    SupervisedContrastiveLoss,
    add_metrics,
    add_projection_head,
    create_classifier,
    create_classifier_imgs_only,
    create_data_augmentation_module,
    create_encoder,
)

# Weights should be loaded from pretext task.

# Image width
image_width = 224

# Images
trainDataDir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_sample/train/"
# trainDataDir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_final/train/"

# Create balanced traing and validation datasets
# batch_size = int(num_classes_per_batch * num_images_per_class)
train_ds = balanced_image_dataset_from_directory(
    trainDataDir,
    num_classes_per_batch=2,
    num_images_per_class=4,
    image_size=(image_width, image_width),
    validation_split=0.2,
    subset="training",
    seed=980801,
    safe_triplet=True,
)

val_ds = balanced_image_dataset_from_directory(
    trainDataDir,
    num_classes_per_batch=2,
    num_images_per_class=4,
    image_size=(image_width, image_width),
    validation_split=0.2,
    subset="validation",
    seed=980801,
    safe_triplet=True,
)

# ------------------------------------------------------
# Modelling
# ------------------------------------------------------
