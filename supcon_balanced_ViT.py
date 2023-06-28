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
    create_vit_encoder,
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
# --- Hyperparameter configuration --- 
input_shape = (image_width, image_width, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100
image_size = 224 
# Size of the patches to be extract from the input images
patch_size = 14
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  
transformer_layers = 8
# Size of the dense layers of the final classifier
mlp_head_units = [512, 256]  
representation_units = 2048
learningRate = 0.001
optimiser = tf.keras.optimizers.Adam(learning_rate=learningRate)
# Classifer
hiddenUnits = 512 
dropoutRate = 0.1
numEpochs = 100
projectionUnits = 128
temperature = 0.05

# Shared functions
# Early stopping on validation loss
callback_EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Image augmentation
data_aug = create_data_augmentation_module()

# ---------------------------------------------------------------------------------
# Baseline classification models using images only
# ---------------------------------------------------------------------------------
print("\nBaseline ViT classification model.\n")
# Setup encoder
encoder = create_vit_encoder(
    data_augmentation=data_aug,
    patch_size = patch_size,
    num_patches = num_patches,
    projection_dim = projection_dim,
    transformer_layers = transformer_layers,
    num_heads = num_heads,
    transformer_units = transformer_units,
    mlp_head_units = mlp_head_units,
    input_shape = input_shape,
    encoder_name = "ViT_encoder",
    representation_units = representation_units)
encoder.summary()
# Setup classifier
classifier = create_classifier_imgs_only(
    encoder_module=encoder,
    model_name="ViT_baseline_classifier",
    input_shape=input_shape,
    hidden_units=hiddenUnits,
    dropout_rate=dropoutRate,
    optimizer=optimiser,
    trainable=True,
)
classifier.summary()
# Define all the callbacks
# Logging
callback_CSVLogger = tf.keras.callbacks.CSVLogger(
    "CSVLogger/train_baseline_ViT.csv"
)
# Training
history = classifier.fit(
    train_ds,
    epochs=numEpochs,
    validation_data=val_ds,
    callbacks=[callback_EarlyStopping, callback_CSVLogger],
    verbose=2,
)
# Save the entire model as a SavedModel.
classifier.save("saved_models/train_baseline_ViT")

# -----------------------------------------------------------------------------------
# Supervised contrastive learning model with images only
# -----------------------------------------------------------------------------------
print(
       "\nSupervised contrastive learning classification model using ViT.\n"
    )
# Pre-train the encoder
encoder = create_vit_encoder(
    data_augmentation=data_aug,
    patch_size = patch_size,
    num_patches = num_patches,
    projection_dim = projection_dim,
    transformer_layers = transformer_layers,
    num_heads = num_heads,
    transformer_units = transformer_units,
    mlp_head_units = mlp_head_units,
    input_shape = input_shape,
    encoder_name = "ViT_encoder_supcon",
    representation_units = representation_units)
encoder.summary()
encoder_with_projection_head = add_projection_head(
    encoder_module=encoder,
    model_name="ViT_encoder_with_projection_head",
    input_shape=input_shape,
    projection_units=projectionUnits,
)
encoder_with_projection_head.compile(
    optimizer=optimiser,
    loss=SupervisedContrastiveLoss(temperature),
)
encoder_with_projection_head.summary()
# Logging
callback_CSVLogger = tf.keras.callbacks.CSVLogger(
    "CSVLogger/supcon_pretrained_encoder_ViT.csv"
)
# Pre-training encoder
history = encoder_with_projection_head.fit(
    train_ds,
    epochs=numEpochs,
    validation_data=val_ds,
    callbacks=[callback_EarlyStopping, callback_CSVLogger],
    verbose=2,
)
# Train the classifier with the frozen encoder
classifier = create_classifier_imgs_only(
    encoder_module=encoder,
    model_name="supcon_classifier_ViT_frozen_encoder",
    input_shape=input_shape,
    hidden_units=hiddenUnits,
    dropout_rate=dropoutRate,
    optimizer=optimiser,
    trainable=False,
)
classifier.summary()
# Logging
callback_CSVLogger = tf.keras.callbacks.CSVLogger(
    "CSVLogger/supcon_encoder_ViT.csv"
)
# Train the classifier with the frozen encoder
history = classifier.fit(
    train_ds,
    epochs=numEpochs,
    validation_data=val_ds,
    callbacks=[callback_EarlyStopping, callback_CSVLogger],
    verbose=2,
)
# Save the entire model as a SavedModel.
classifier.save(f"saved_models/supcon_encoder_ViT")