"""Supervised contrastive learning using two CNN encoder types image data, with perfectly balanced batching."""
import csv
import tensorflow as tf

from functions import (
    load_generators,
    SupervisedContrastiveLoss,
    add_metrics,
    add_projection_head,
    create_classifier_imgs_only,
    create_data_augmentation_module,
    create_encoder,
    find_optimal_threshold,
)

# Image width
image_width = 224

# Data directories
trainDataDir = "local_directory/train/"
validDataDir = "local_directory/valid/"
testDataDir = "local_directory/test/"

train_ds, val_ds, valid_ds_unbalanced, test_ds = load_generators(
    trainDataDir=trainDataDir,
    validDataDir=validDataDir,
    testDataDir=testDataDir,
    image_width=image_width,
    num_images_per_class=24,
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
        train_ds,
        epochs=numEpochs,
        validation_data=val_ds,
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )

    # Evaluate
    print("Evaluate on test data")
    test_predictions, optimal_threshold, test_metrics = find_optimal_threshold(
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
        f"supcon_malignant_repo/CSVLogger/supcon_{enc}.csv"
    )
    # Train the classifier with the frozen encoder
    history = classifier.fit(
        train_ds,
        epochs=numEpochs,
        validation_data=val_ds,
        callbacks=[callback_EarlyStopping, callback_CSVLogger],
        verbose=2,
    )

    # Evaluate
    print("Evaluate on test data")
    test_predictions, optimal_threshold, test_metrics = find_optimal_threshold(
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


# -----------------------------------------------------------------------------------
# Adding metrics
# -----------------------------------------------------------------------------------

# Add metrics to csv file
# Training
add_metrics(
    hist_filelocation="supcon_malignant_repo/CSVLogger/supcon_ResNet50V2.csv",
    saved_name="supcon_malignant_repo/CSVLogger/supcon_ResNet50V2_added_metrics.csv",
)
add_metrics(
    hist_filelocation="supcon_malignant_repo/CSVLogger/train_baseline_ResNet50V2.csv",
    saved_name="supcon_malignant_repo/CSVLogger/train_baseline_ResNet50V2_added_metrics.csv",
)
add_metrics(
    hist_filelocation="supcon_malignant_repo/CSVLogger/supcon_InceptionV3.csv",
    saved_name="supcon_malignant_repo/CSVLogger/supcon_InceptionV3_added_metrics.csv",
)
add_metrics(
    hist_filelocation="supcon_malignant_repo/CSVLogger/train_baseline_InceptionV3.csv",
    saved_name="supcon_malignant_repo/CSVLogger/train_baseline_InceptionV3_added_metrics.csv",
)

# -----------------------------------------------------------------------------------
# Plotting is in another python file
# -----------------------------------------------------------------------------------
