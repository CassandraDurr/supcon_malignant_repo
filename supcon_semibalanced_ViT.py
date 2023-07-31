"""Supervised contrastive learning using two CNN encoder types image data, with balanced batching."""
import csv
import os
import tensorflow as tf

from functions import (
    custom_data_generator,
    find_optimal_threshold,
    SupervisedContrastiveLoss,
    add_metrics,
    add_projection_head,
    create_classifier_imgs_only,
    create_data_augmentation_module,
    create_vit_encoder,
)

# Weights should be loaded from pretext task.

# Image width, batch size
image_width = 224
batch_size = 48 

# Data directories
trainDataDir = "local_directory/train/"
validDataDir = "local_directory/valid/"
testDataDir = "local_directory/test/"

# Create balanced training and validation datasets
train_generator = custom_data_generator(data_dir=trainDataDir, batch_size = batch_size, class_0_ratio=0.8, image_width=image_width)
validation_generator = custom_data_generator(data_dir=validDataDir, batch_size = batch_size, class_0_ratio=0.8, image_width=image_width)

# Calculate the number of steps per epoch and validation steps
train_steps_per_epoch = len(os.listdir(os.path.join(trainDataDir, '0'))) + len(os.listdir(os.path.join(trainDataDir, '1')))
val_steps = len(os.listdir(os.path.join(validDataDir, '0'))) + len(os.listdir(os.path.join(validDataDir, '1')))

# Unbalanced test and validation datasets
# Create an ImageDataGenerator for testing data (don't need balanced test sets)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = test_datagen.flow_from_directory(
    testDataDir,
    target_size=(image_width, image_width), 
    batch_size=batch_size,
    class_mode='binary',     
    shuffle=False            
)
# To select the optimal threshold we will also create a data generator 
valid_datagen_unbalanced = tf.keras.preprocessing.image.ImageDataGenerator()
valid_ds_unbalanced = valid_datagen_unbalanced.flow_from_directory(
    validDataDir,
    target_size=(image_width, image_width), 
    batch_size=batch_size,
    class_mode='binary',     
    shuffle=False            
)

# ------------------------------------------------------
# Modelling
# ------------------------------------------------------
# --- Hyperparameter configuration ---
input_shape = (image_width, image_width, 3)
learning_rate = 0.001
weight_decay = 0.0001
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
optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# Classifer
hiddenUnits = 512
dropoutRate = 0.1
numEpochs = 100
projectionUnits = 128
temperature = 0.1

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
    weight_location="supcon_malignant_repo/saved_models/encoder_weights_ViT_MSE.h5",
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
callback_CSVLogger = tf.keras.callbacks.CSVLogger(
    "supcon_malignant_repo/CSVLogger/train_baseline_ViT.csv"
)
# Training
history = classifier.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch // batch_size,
    epochs=numEpochs,
    validation_data=validation_generator,
    validation_steps=val_steps // batch_size,
    callbacks=[callback_EarlyStopping, callback_CSVLogger],
    verbose=2,
)

# Save the entire model as a SavedModel.
# classifier.save("saved_models/train_baseline_ViT")

# Evaluate
print("Evaluate on test data")
csv_file = "supcon_malignant_repo/CSVLogger/test_baseline_ViT.csv"
test_predictions, optimal_threshold, test_metrics = find_optimal_threshold(
        classifier=classifier, valid_dataset=valid_ds_unbalanced, test_dataset=test_ds
    )
with open(csv_file, mode="w", newline="") as file:
        fieldnames = [
            "Threshold",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(
            {
                "Threshold": optimal_threshold,
                "AUC": test_metrics["auc"],
                "Accuracy": test_metrics["accuracy"],
                "Precision": test_metrics["precision"],
                "Recall": test_metrics["recall"],
                "Specificity": test_metrics["specificity"],
                "F1": test_metrics["f1"],
            }
        )

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
    weight_location="supcon_malignant_repo/saved_models/encoder_weights_ViT_MSE.h5",
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
    "supcon_malignant_repo/CSVLogger/supcon_pretrained_encoder_ViT.csv"
)
# Pre-training encoder
history = encoder_with_projection_head.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch // batch_size,
    epochs=numEpochs,
    validation_data=validation_generator,
    validation_steps=val_steps // batch_size,
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
    "supcon_malignant_repo/CSVLogger/supcon_encoder_ViT.csv"
)
# Train the classifier with the frozen encoder
history = classifier.fit(
    train_generator,
    steps_per_epoch=train_steps_per_epoch // batch_size,
    epochs=numEpochs,
    validation_data=validation_generator,
    validation_steps=val_steps // batch_size,
    callbacks=[callback_EarlyStopping, callback_CSVLogger],
    verbose=2,
)

# Save the entire model as a SavedModel.
# classifier.save("saved_models/supcon_encoder_ViT")

# Evaluate
print("Evaluate on test data")
test_predictions, optimal_threshold, test_metrics = find_optimal_threshold(
        classifier=classifier, valid_dataset=valid_ds_unbalanced, test_dataset=test_ds
    )
csv_file = "supcon_malignant_repo/CSVLogger/test_supcon_ViT.csv"
with open(csv_file, mode="w", newline="") as file:
        fieldnames = [
            "Threshold",
            "AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(
            {
                "Threshold": optimal_threshold,
                "AUC": test_metrics["auc"],
                "Accuracy": test_metrics["accuracy"],
                "Precision": test_metrics["precision"],
                "Recall": test_metrics["recall"],
                "Specificity": test_metrics["specificity"],
                "F1": test_metrics["f1"],
            }
        )

# -----------------------------------------------------------------------------------
# Adding metrics
# -----------------------------------------------------------------------------------

# Add metrics to csv file
add_metrics(
    hist_filelocation="supcon_malignant_repo/CSVLogger/supcon_encoder_ViT.csv",
    saved_name="supcon_malignant_repo/CSVLogger/supcon_encoder_ViT_added_metrics.csv",
)
add_metrics(
    hist_filelocation="supcon_malignant_repo/CSVLogger/train_baseline_ViT.csv",
    saved_name="supcon_malignant_repo/CSVLogger/train_baseline_ViT_added_metrics.csv",
)

