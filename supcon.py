"""Supervised contrastive learning using two encoder types and either tabular and image data or just image data."""
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from functions import (SupervisedContrastiveLoss, add_metrics,
                       add_projection_head, create_classifier,
                       create_classifier_imgs_only,
                       create_data_augmentation_module, create_encoder)

# Weights should be loaded from pretext task.

# Full training data
train_df = pd.read_csv("Training_data_ready_with_target_and_ID.csv")
train_df.drop(columns="Unnamed: 0", inplace=True)

# Get list of images in training dataset
trainDataDir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_sample/train/"
# trainDataDir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_final/train/"
image_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    trainDataDir,
    shuffle=False,
    class_mode="binary",
    target_size=(224, 224),
    batch_size=400,
)  # total amount of 0 and 1 values
images, labels = next(image_generator)

# Getting the ordered list of filenames for the images
image_files = pd.Series(image_generator.filenames)
# Form = 0\ISIC_0015719.jpg
image_files = image_files.str.split("\\", expand=True)[1].str[:-4]
# Form = ISIC_0015719
image_files = list(image_files)

# Now reduce the training data df to reflect chosen data
train_df = train_df[train_df["image_name"].isin(image_files)]
# Check the order of records in the training data vs image_files
if image_files == list(train_df.image_name):
    print("The lists are identical")
    # Drop image name
    train_df.drop(columns=["image_name"], inplace=True)
else:
    print("The lists are not identical")
    sys.exit()

# Set up the train/ validation split
(train_tabular, valid_tabular, train_images, valid_images) = train_test_split(
    train_df, images, test_size=0.2, random_state=42
)  # 33,126 images initially
print(f"Training tabular data shape = {train_tabular.shape}")
print(f"Validation tabular data shape = {valid_tabular.shape}")
print(f"Training image data shape = {train_images.shape}")
print(f"Validation image data shape = {valid_images.shape}")

# Set aside the target
train_target = train_tabular["target"]
train_target = np.array(train_target)
print(f"Train target shape = {train_target.shape}")
valid_target = valid_tabular["target"]
valid_target = np.array(valid_target)
print(f"Validation target shape = {valid_target.shape}")

# Now drop target from the original dataframes
train_tabular.drop(columns=["target"], inplace=True)
valid_tabular.drop(columns=["target"], inplace=True)

# ------------------------------------------------------
# Modelling
# ------------------------------------------------------
# Shared parameters
numClasses = 2
input_shape = (224, 224, 3)
learningRate = 0.001
batchSize = 6  # 256
hiddenUnits = 512
projectionUnits = 128
numEpochs = 100
dropoutRate = 0.1
temperature = 0.05
optimiser = tf.keras.optimizers.Adam(learning_rate=learningRate)

# Shared functions
# Early stopping on validation loss
callback_EarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Image augmentation
data_aug = create_data_augmentation_module()
# Setting the state of the normalization layer.
data_aug.layers[0].adapt(train_images)

# ---------------------------------------------------------------------------------
# Baseline classification models using images only
# ---------------------------------------------------------------------------------
encoder_types = ["ResNet50V2", "InceptionV3"]
for enc in encoder_types:
    print(f"\nBaseline classification model using {enc}, only images\n")
    # Setup encoder
    encoder = create_encoder(
        encoder_type=enc,
        input_shape=input_shape,
        data_augmentation=data_aug,
        encoder_name=f"{enc}_encoder",
        encoder_weights_location=f"saved_models/encoder_weights_{enc}_MSE.h5",
    )
    encoder.summary()
    # Setup classifier
    classifier = create_classifier_imgs_only(
        encoder_module=encoder,
        model_name=f"{enc}_baseline_classifier",
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
        f"CSVLogger/train_baseline_{enc}.csv"
    )
    # Training
    history = classifier.fit(
        x=train_images,
        y=train_target.reshape(
            -1,
        ),
        batch_size=batchSize,
        epochs=numEpochs,
        validation_data=[
            valid_images,
            valid_target.reshape(
                -1,
            ),
        ],
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )
    # Save the entire model as a SavedModel.
    classifier.save(f"saved_models/train_baseline_{enc}")


# -----------------------------------------------------------------------------------
# Supervised contrastive learning model with images only
# -----------------------------------------------------------------------------------
for enc in encoder_types:
    print(
        f"\nSupervised contrastive learning classification model using {enc}, only images\n"
    )
    # Pre-train the encoder
    encoder = create_encoder(
        encoder_type=enc,
        input_shape=input_shape,
        data_augmentation=data_aug,
        encoder_name=f"{enc}_encoder_supcon",
        encoder_weights_location=f"saved_models/encoder_weights_{enc}_MSE.h5",
    )
    encoder.summary()
    encoder_with_projection_head = add_projection_head(
        encoder_module=encoder,
        model_name=f"{enc}_encoder_with_projection_head",
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
        f"CSVLogger/supcon_pretrained_encoder_{enc}.csv"
    )
    # Pre-training encoder
    history = encoder_with_projection_head.fit(
        x=train_images,
        y=train_target.reshape(
            -1,
        ),
        batch_size=batchSize,
        epochs=numEpochs,
        validation_data=[
            valid_images,
            valid_target.reshape(
                -1,
            ),
        ],
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )
    # Train the classifier with the frozen encoder
    classifier = create_classifier_imgs_only(
        encoder_module=encoder,
        model_name=f"supcon_classifier_{enc}_frozen_encoder",
        input_shape=input_shape,
        hidden_units=hiddenUnits,
        dropout_rate=dropoutRate,
        optimizer=optimiser,
        trainable=False,
    )
    classifier.summary()
    # Logging
    callback_CSVLogger = tf.keras.callbacks.CSVLogger(
        f"CSVLogger/supcon_encoder_{enc}.csv"
    )
    # Train the classifier with the frozen encoder
    history = classifier.fit(
        x=train_images,
        y=train_target.reshape(
            -1,
        ),
        batch_size=batchSize,
        epochs=numEpochs,
        validation_data=[
            valid_images,
            valid_target.reshape(
                -1,
            ),
        ],
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )
    # Save the entire model as a SavedModel.
    classifier.save(f"saved_models/supcon_encoder_{enc}")


# ---------------------------------------------------------------------------------
# Baseline classification models using image and tabular data
# ---------------------------------------------------------------------------------
for enc in encoder_types:
    print(f"\nBaseline classification model using {enc}, images and data\n")
    # Setup encoder
    encoder = create_encoder(
        encoder_type=enc,
        input_shape=input_shape,
        data_augmentation=data_aug,
        encoder_name=f"{enc}_encoder",
        encoder_weights_location=f"saved_models/encoder_weights_{enc}_MSE.h5",
    )
    encoder.summary()
    # Setup classifier
    classifier = create_classifier(
        encoder_module=encoder,
        model_name=f"{enc}_baseline_classifier",
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
        f"CSVLogger/train_baseline_{enc}_incl_tabular.csv"
    )
    # Training
    history = classifier.fit(
        x=[train_images, train_tabular],
        y=np.array(train_target).reshape(-1, 1),
        batch_size=batchSize,
        epochs=numEpochs,
        validation_data=[
            [valid_images, valid_tabular],
            np.array(valid_target).reshape(-1, 1),
        ],
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )
    # Save the entire model as a SavedModel.
    classifier.save(f"saved_models/train_baseline_{enc}_incl_tabular")


# -----------------------------------------------------------------------------------
# Supervised contrastive learning model with image and tabular data
# -----------------------------------------------------------------------------------
for enc in encoder_types:
    print(f"\nSupervised contrastive learning classification model using {enc}\n")
    # Pre-train the encoder
    encoder = create_encoder(
        encoder_type=enc,
        input_shape=input_shape,
        data_augmentation=data_aug,
        encoder_name=f"{enc}_encoder_supcon",
        encoder_weights_location=f"saved_models/encoder_weights_{enc}_MSE.h5",
    )
    encoder.summary()
    encoder_with_projection_head = add_projection_head(
        encoder_module=encoder,
        model_name=f"{enc}_encoder_with_projection_head",
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
        f"CSVLogger/supcon_pretrained_encoder_{enc}_incl_tabular.csv"
    )
    # Pre-training encoder
    history = encoder_with_projection_head.fit(
        x=train_images,
        y=np.array(train_target).reshape(-1, 1),
        batch_size=batchSize,
        epochs=numEpochs,
        validation_data=[valid_images, np.array(valid_target).reshape(-1, 1)],
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )
    # Train the classifier with the frozen encoder
    classifier = create_classifier(
        encoder_module=encoder,
        model_name=f"supcon_classifier_{enc}_frozen_encoder",
        input_shape=input_shape,
        hidden_units=hiddenUnits,
        dropout_rate=dropoutRate,
        optimizer=optimiser,
        trainable=False,
    )
    classifier.summary()
    # Logging
    callback_CSVLogger = tf.keras.callbacks.CSVLogger(
        f"CSVLogger/supcon_encoder_{enc}_incl_tabular.csv"
    )
    # Train the classifier with the frozen encoder
    history = classifier.fit(
        x=[train_images, train_tabular],
        y=np.array(train_target).reshape(-1, 1),
        batch_size=batchSize,
        epochs=numEpochs,
        validation_data=[
            [valid_images, valid_tabular],
            np.array(valid_target).reshape(-1, 1),
        ],
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )
    # Save the entire model as a SavedModel.
    classifier.save(f"saved_models/supcon_encoder_{enc}_incl_tabular")

# -----------------------------------------------------------------------------------
# Adding metrics
# -----------------------------------------------------------------------------------

# Add metrics to csv file
add_metrics(
    hist_filelocation="CSVLogger/train_baseline_InceptionV3_incl_tabular.csv",
    saved_name="CSVLogger/train_baseline_InceptionV3_incl_tabular_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/train_baseline_ResNet50V2_incl_tabular.csv",
    saved_name="CSVLogger/train_baseline_ResNet50V2_incl_tabular_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/supcon_encoder_InceptionV3_incl_tabular.csv",
    saved_name="CSVLogger/supcon_encoder_InceptionV3_incl_tabular_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/supcon_encoder_ResNet50V2_incl_tabular.csv",
    saved_name="CSVLogger/supcon_encoder_ResNet50V2_incl_tabular_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/supcon_encoder_InceptionV3.csv",
    saved_name="CSVLogger/supcon_encoder_InceptionV3_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/supcon_encoder_ResNet50V2.csv",
    saved_name="CSVLogger/supcon_encoder_ResNet50V2_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/supcon_encoder_InceptionV3.csv",
    saved_name="CSVLogger/supcon_encoder_InceptionV3_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/train_baseline_InceptionV3.csv",
    saved_name="CSVLogger/train_baseline_InceptionV3_added_metrics.csv",
)

# -----------------------------------------------------------------------------------
# Plotting is in another python file
# -----------------------------------------------------------------------------------
