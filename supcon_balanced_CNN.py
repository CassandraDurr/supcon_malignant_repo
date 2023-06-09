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
    create_encoder,
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
# Shared parameters
numClasses = 2
input_shape = (image_width, image_width, 3)
learningRate = 0.001
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
        train_ds,
        epochs=numEpochs,
        validation_data=val_ds,
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )
    # Save the entire model as a SavedModel.
    classifier.save(f"saved_models/train_baseline_{enc}")
    
    # Evaluate
    print("Evaluate on test data")
    results = classifier.evaluate(test_ds, verbose=2)
    
    # Write evaluation results to CSV file
    csv_file = f"CSVLogger/test_baseline_{enc}.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write column names
        writer.writerow(classifier.metrics_names)  
        # Write evaluation results
        writer.writerow(results)  


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
        f"CSVLogger/supcon_pretraining_{enc}.csv"
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
        f"CSVLogger/supcon_{enc}.csv"
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
    # classifier.save(f"saved_models/supcon_{enc}")
    
    # Testing
    # Evaluate
    print("Evaluate on test data")
    results = classifier.evaluate(test_ds, verbose=2)
    
    # Write evaluation results to CSV file
    csv_file = f"CSVLogger/test_supcon_{enc}.csv"
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
# Training
add_metrics(
    hist_filelocation="CSVLogger/supcon_InceptionV3.csv",
    saved_name="CSVLogger/supcon_InceptionV3_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/supcon_ResNet50V2.csv",
    saved_name="CSVLogger/supcon_ResNet50V2_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/train_baseline_ResNet50V2.csv",
    saved_name="CSVLogger/train_baseline_ResNet50V2_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/train_baseline_InceptionV3.csv",
    saved_name="CSVLogger/train_baseline_InceptionV3_added_metrics.csv",
)
# Testing
add_metrics(
    hist_filelocation="CSVLogger/test_supcon_InceptionV3.csv",
    saved_name="CSVLogger/test_supcon_InceptionV3_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/test_baseline_InceptionV3.csv",
    saved_name="CSVLogger/test_baseline_InceptionV3_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/test_supcon_ResNet50V2.csv",
    saved_name="CSVLogger/test_supcon_ResNet50V2_added_metrics.csv",
)
add_metrics(
    hist_filelocation="CSVLogger/test_baseline_ResNet50V2.csv",
    saved_name="CSVLogger/test_baseline_ResNet50V2_added_metrics.csv",
)

# -----------------------------------------------------------------------------------
# Plotting is in another python file
# -----------------------------------------------------------------------------------
