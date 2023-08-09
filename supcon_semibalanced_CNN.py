"""Supervised contrastive learning using two CNN encoder types image data, with semi-balanced batching."""
import csv
import os
import tensorflow as tf

from functions import (
    custom_data_generator,
    SupervisedContrastiveLoss,
    add_metrics,
    add_projection_head,
    create_classifier_imgs_only,
    create_data_augmentation_module,
    create_encoder,
    find_optimal_threshold,
)

# Image width, batch size
image_width = 224
batch_size = 16

# Data directories
trainDataDir = "D:/HAM10000/train/"
validDataDir = "D:/HAM10000/valid/"
testDataDir = "D:/HAM10000/test/"

# Create balanced training and validation datasets
train_generator = custom_data_generator(
    data_dir=trainDataDir,
    batch_size=batch_size,
    class_0_ratio=0.75,
    image_width=image_width,
)
validation_generator = custom_data_generator(
    data_dir=validDataDir,
    batch_size=batch_size,
    class_0_ratio=0.75,
    image_width=image_width,
)

# Calculate the number of steps per epoch and validation steps
train_steps_per_epoch = len(os.listdir(os.path.join(trainDataDir, "0"))) + len(
    os.listdir(os.path.join(trainDataDir, "1"))
)
val_steps = len(os.listdir(os.path.join(validDataDir, "0"))) + len(
    os.listdir(os.path.join(validDataDir, "1"))
)

# Unbalanced test and validation datasets
# Create an ImageDataGenerator for testing data (don't need balanced test sets)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_ds = test_datagen.flow_from_directory(
    testDataDir,
    target_size=(image_width, image_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
)
# To select the optimal threshold we will also create a data generator
valid_datagen_unbalanced = tf.keras.preprocessing.image.ImageDataGenerator()
valid_ds_unbalanced = valid_datagen_unbalanced.flow_from_directory(
    validDataDir,
    target_size=(image_width, image_width),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
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
temperature = 0.1
optimiser = tf.keras.optimizers.Adam(learning_rate=learningRate)

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
encoder_types = ["InceptionV3"]
# encoder_types = ["ResNet50V2", "InceptionV3"]
for enc in encoder_types:
    print(f"\nBaseline classification model using {enc}, only images\n")
    # Setup encoder
    encoder = create_encoder(
        encoder_type=enc,
        input_shape=input_shape,
        data_augmentation=data_aug,
        encoder_name=f"{enc}_encoder",
        encoder_weights_location=f"supcon_malignant_repo/saved_models/encoder_weights_{enc}_MSE.h5",
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
        f"supcon_malignant_repo/CSVLogger/train_baseline_{enc}.csv"
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

    # Evaluate
    print("Evaluate on test data")
    test_predictions, optimal_threshold, test_metrics, valid_metrics = find_optimal_threshold(
        classifier=classifier, valid_dataset=valid_ds_unbalanced, test_dataset=test_ds
    )

    # Write evaluation results to CSV file
    csv_file = f"supcon_malignant_repo/CSVLogger/test_baseline_{enc}.csv"
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
        
    # Evaluate on validation data
    with open(csv_file.replace("test", "valid"), mode="w", newline="") as file:
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
                "AUC": valid_metrics["auc"],
                "Accuracy": valid_metrics["accuracy"],
                "Precision": valid_metrics["precision"],
                "Recall": valid_metrics["recall"],
                "Specificity": valid_metrics["specificity"],
                "F1": valid_metrics["f1"],
            }
        )


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
        encoder_weights_location=f"supcon_malignant_repo/saved_models/encoder_weights_{enc}_MSE.h5",
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
        f"supcon_malignant_repo/CSVLogger/supcon_pretraining_{enc}.csv"
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
        f"supcon_malignant_repo/CSVLogger/supcon_{enc}.csv"
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

    # Evaluate
    print("Evaluate on test data")
    test_predictions, optimal_threshold, test_metrics, valid_metrics = find_optimal_threshold(
        classifier=classifier, valid_dataset=valid_ds_unbalanced, test_dataset=test_ds
    )

    # Write evaluation results to CSV file
    csv_file = f"supcon_malignant_repo/CSVLogger/test_supcon_{enc}.csv"
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
        
    # Evaluate on validation data
    with open(csv_file.replace("test", "valid"), mode="w", newline="") as file:
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
                "AUC": valid_metrics["auc"],
                "Accuracy": valid_metrics["accuracy"],
                "Precision": valid_metrics["precision"],
                "Recall": valid_metrics["recall"],
                "Specificity": valid_metrics["specificity"],
                "F1": valid_metrics["f1"],
            }
        )


    

# -----------------------------------------------------------------------------------
# Adding metrics
# -----------------------------------------------------------------------------------

# Add metrics to csv file
for enc in encoder_types:
    add_metrics(
        hist_filelocation=f"supcon_malignant_repo/CSVLogger/supcon_{enc}.csv",
        saved_name=f"supcon_malignant_repo/CSVLogger/supcon_{enc}_added_metrics.csv",
    )
    add_metrics(
        hist_filelocation=f"supcon_malignant_repo/CSVLogger/train_baseline_{enc}.csv",
        saved_name=f"supcon_malignant_repo/CSVLogger/train_baseline_{enc}_added_metrics.csv",
    )
