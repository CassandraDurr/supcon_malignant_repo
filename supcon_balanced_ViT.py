"""Supervised contrastive learning using two CNN encoder types image data, with balanced batching."""
import csv
import tensorflow as tf
from kerasgen.balanced_image_dataset import balanced_image_dataset_from_directory

from functions import (
    SupervisedContrastiveLoss,
    add_metrics,
    add_projection_head,
    create_classifier_imgs_only,
    create_data_augmentation_module,
    create_vit_encoder,
)

# Weights should be loaded from pretext task.

# Image width
image_width = 224

# Images
trainDataDir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_final/train/"
testDataDir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_final/test/"
# trainDataDir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_final/train/"

# Create balanced traing and validation datasets
# batch_size = int(num_classes_per_batch * num_images_per_class)
train_ds = balanced_image_dataset_from_directory(
    trainDataDir,
    num_classes_per_batch=2,
    num_images_per_class=8,
    image_size=(image_width, image_width),
    validation_split=0.2,
    subset="training",
    seed=980801,
    safe_triplet=True,
)

val_ds = balanced_image_dataset_from_directory(
    trainDataDir,
    num_classes_per_batch=2,
    num_images_per_class=8,
    image_size=(image_width, image_width),
    validation_split=0.2,
    subset="validation",
    seed=980801,
    safe_triplet=True,
)

# Create an ImageDataGenerator for testing data
# We don't need balanced test sets
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Create a test generator using the test directory
test_ds = test_datagen.flow_from_directory(
    testDataDir,
    target_size=(image_width, image_width), 
    batch_size=16,
    class_mode='binary',     # Binary classification problem
    shuffle=False            # Disable shuffling to maintain order for evaluation
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
data_aug = create_data_augmentation_module(RandomRotationAmount=0.5)

# ---------------------------------------------------------------------------------
# Baseline classification models using images only
# ---------------------------------------------------------------------------------
print("\nBaseline ViT classification model.\n")
# Setup encoder
encoder = create_vit_encoder(
    data_augmentation=data_aug,
    patch_size=patch_size,
    num_patches=num_patches,
    projection_dim=projection_dim,
    transformer_layers=transformer_layers,
    num_heads=num_heads,
    transformer_units=transformer_units,
    input_shape=input_shape,
    encoder_name="ViT_encoder",
    representation_units=representation_units,
    load_weights=True,
    weight_location="saved_models/encoder_weights_ViT_MSE.h5",
)
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
callback_CSVLogger = tf.keras.callbacks.CSVLogger("CSVLogger/train_baseline_ViT.csv")
# Training
history = classifier.fit(
    train_ds,
    epochs=numEpochs,
    validation_data=val_ds,
    callbacks=[callback_EarlyStopping, callback_CSVLogger],
    verbose=2,
)

# Save the entire model as a SavedModel.
# classifier.save("saved_models/train_baseline_ViT")

# Evaluate
print("Evaluate on test data")
results = classifier.evaluate(test_ds, verbose=2)

# Write evaluation results to CSV file
csv_file = "CSVLogger/test_baseline_ViT.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write column names
    writer.writerow(classifier.metrics_names)  
    # Write evaluation results
    writer.writerow(results) 

# -----------------------------------------------------------------------------------
# Supervised contrastive learning model with images only
# -----------------------------------------------------------------------------------
print("\nSupervised contrastive learning classification model using ViT.\n")
# Pre-train the encoder
encoder = create_vit_encoder(
    data_augmentation=data_aug,
    patch_size=patch_size,
    num_patches=num_patches,
    projection_dim=projection_dim,
    transformer_layers=transformer_layers,
    num_heads=num_heads,
    transformer_units=transformer_units,
    input_shape=input_shape,
    encoder_name="ViT_encoder_supcon",
    representation_units=representation_units,
    load_weights=True,
    weight_location="saved_models/encoder_weights_ViT_MSE.h5",
)
encoder.summary()

# Add projection head
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
callback_CSVLogger = tf.keras.callbacks.CSVLogger("CSVLogger/supcon_encoder_ViT.csv")
# Train the classifier with the frozen encoder
history = classifier.fit(
    train_ds,
    epochs=numEpochs,
    validation_data=val_ds,
    callbacks=[callback_EarlyStopping, callback_CSVLogger],
    verbose=2,
)

# Save the entire model as a SavedModel.
# classifier.save("saved_models/supcon_encoder_ViT")

# Evaluate
print("Evaluate on test data")
results = classifier.evaluate(test_ds, verbose=2)

# Write evaluation results to CSV file
csv_file = "CSVLogger/test_supcon_ViT.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write column names
    writer.writerow(classifier.metrics_names)  
    # Write evaluation results
    writer.writerow(results) 

# -----------------------------------------------------------------------------------
# Adding metrics
# -----------------------------------------------------------------------------------

# Add metrics to csv file
add_metrics(
    hist_filelocation="CSVLogger/supcon_encoder_ViT.csv",
    saved_name="CSVLogger/supcon_encoder_ViT_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/train_baseline_ViT.csv",
    saved_name="CSVLogger/train_baseline_ViT_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/test_supcon_ViT.csv",
    saved_name="CSVLogger/test_supcon_ViT_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/test_baseline_ViT.csv",
    saved_name="CSVLogger/test_baseline_ViT_added_metrics.csv",
)
