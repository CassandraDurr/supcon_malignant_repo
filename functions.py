import re
import tensorflow as tf
import tensorflow_addons as tfa
import plotly.graph_objects as go
import numpy as np
import pandas as pd

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


# Create encoder function
def create_encoder(
    encoder_type: str,
    input_shape: tuple,
    data_augmentation: tf.keras.Sequential,
    encoder_name: str,
    encoder_weights_location: str,
) -> tf.keras.Model:
    """This function creates a TensorFlow encoder model without trained weights.

    The encoder is either "ResNet50V2" or "InceptionV3".

    Args:
        encoder_type (str): The type of encoder to be used, either "ResNet50V2" or "InceptionV3".
        input_shape (tuple): Shape of the image data (width, height, channels).
        data_augmentation (tf.keras.Sequential): A tensorflow sequential model performing data augmentation.
        encoder_name (str): The name of the encoder model to be returned.
        encoder_weights_location (str): String location of the encoder weights.

    Returns:
        tf.keras.Model: A TensorFlow encoder model.
    """
    # Build encoder module
    if encoder_type == "ResNet50V2":
        encoder_module = tf.keras.applications.ResNet50V2(
            include_top=False, weights=None, input_shape=input_shape
        )
        encoder_module.load_weights(encoder_weights_location)
    elif encoder_type == "InceptionV3":
        encoder_module = tf.keras.applications.InceptionV3(
            include_top=False, weights=None, input_shape=input_shape
        )
        encoder_module.load_weights(encoder_weights_location)
    else:
        raise NotImplementedError(
            "This function only produces ResNet50V2 or InceptionV3 encoders."
        )
    # Create model with the chosen encoder
    input_module = tf.keras.Input(shape=input_shape)
    augmentation_module = data_augmentation(input_module)
    output_module = encoder_module(augmentation_module)
    output_module = tf.keras.layers.GlobalAveragePooling2D()(output_module)
    model = tf.keras.Model(
        inputs=input_module, outputs=output_module, name=encoder_name
    )
    return model


# Create data-augmentation module
def create_data_augmentation_module(
    RandomRotationAmount: float = 0.02,
) -> tf.keras.Sequential:
    """This function creates a data augmentation module for TensorFlow models.

    Args:
        RandomRotationAmount (float, optional): Argument for tf.keras.layers.RandomRotation. Defaults to 0.02.

    Returns:
        tf.keras.Sequential: Data augmentation module
    """
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.Normalization(),
            tf.keras.layers.GaussianNoise(stddev=0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(RandomRotationAmount)
            # AttributeError: module 'tensorflow.keras.layers' has no attribute 'RandomBrightness'
            # tf.keras.layers.RandomBrightness(RandomBrightnessAmount),
        ]
    )
    return data_augmentation


# Supervised Contrastive Loss function from paper by Prannay Khosla et al.
# https://keras.io/examples/vision/supervised-contrastive-learning/
class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


# Incorporate projection head for the pre-training of the encoder with SupervisedContrastiveLoss
def add_projection_head(
    encoder_module: tf.keras.Model,
    model_name: str,
    input_shape: tuple,
    projection_units: int,
) -> tf.keras.Model:
    """Figure out what the projection head does.

    Args:
        encoder_module (tf.keras.Model): TensorFlow encoder module.
        model_name (str): Name of the model with added projection head.
        input_shape (tuple): Shape of the image data (width, height, channels).
        projection_units (int): The number of units of the dense layer/ projection head.

    Returns:
        tf.keras.Model: TensorFlow encoder model with a projection head.
    """
    input_module = tf.keras.Input(shape=input_shape)
    encoder = encoder_module(input_module)
    # Include dense layer/ projection head
    output_module = tf.keras.layers.Dense(projection_units, activation="relu")(encoder)
    model = tf.keras.Model(inputs=input_module, outputs=output_module, name=model_name)
    return model


# Create a classifier for the images only models
def create_classifier_imgs_only(
    encoder_module: tf.keras.Model,
    model_name: str,
    input_shape: tuple,
    hidden_units: int,
    dropout_rate: float,
    optimizer: tf.keras.optimizers.Adam,
    trainable: bool = True,
) -> tf.keras.Model:
    """Classification model made up of a encoder and MLP for image data only.

    Args:
        encoder_module (tf.keras.Model): TensorFlow encoder module.
        input_shape (tuple): Shape of the image data (width, height, channels).
        hidden_units (int): Number of units in the MLP dense layer after the encoder.
        dropout_rate (float): Rate of dropout in dropout layers in MLP after the encoder.
        optimizer (tf.keras.optimizers.Adam): Optimiser for model.
        trainable (bool, optional): Whether the encoder layers are trainable or not (frozen). Defaults to True.

    Returns:
        tf.keras.Model: Model with encoder and classifier.
    """

    # Set encoder layers to be frozen or trainable
    for layer in encoder_module.layers:
        layer.trainable = trainable
    # Set up layers for the model
    input_module = tf.keras.Input(shape=input_shape)
    features = encoder_module(input_module)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    output_module = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    # Build module
    model = tf.keras.Model(inputs=input_module, outputs=output_module, name=model_name)
    # Metrics for binary classification
    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives(),
    ]
    # Compile model using BinaryFocalCrossentropy
    model.compile(
        loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=metrics,
    )
    return model


# Create a classifier for the images only models
def create_classifier(
    encoder_module: tf.keras.Model,
    model_name: str,
    input_shape: tuple,
    hidden_units: int,
    dropout_rate: float,
    optimizer: tf.keras.optimizers.Adam,
    trainable: bool = True,
) -> tf.keras.Model:
    """Classification model made up of a encoder and MLP.

    Args:
        encoder_module (tf.keras.Model): TensorFlow encoder module.
        input_shape (tuple): Shape of the image data (width, height, channels).
        hidden_units (int): Number of units in the MLP dense layer after the encoder.
        dropout_rate (float): Rate of dropout in dropout layers in MLP after the encoder.
        optimizer (tf.keras.optimizers.Adam): Optimiser for model.
        trainable (bool, optional): Whether the encoder layers are trainable or not (frozen). Defaults to True.

    Returns:
        tf.keras.Model: Model with encoder and classifier.
    """

    # Set encoder layers to be frozen or trainable
    for layer in encoder_module.layers:
        layer.trainable = trainable
    # Set up layers for the model - images
    input_module = tf.keras.Input(shape=input_shape)
    features = encoder_module(input_module)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
    features = tf.keras.layers.Dropout(dropout_rate)(features)
    flatten = tf.keras.layers.Flatten()(features)
    flatten = tf.keras.layers.Dense(64, activation="relu")(flatten)
    # Set up layers for the model - tabular
    model_input_tabular = tf.keras.layers.Input(shape=(8,), name="input_tabular")
    tabular_dense = tf.keras.layers.Dense(8, activation="relu")(model_input_tabular)
    tabular_dense = tf.keras.layers.Dropout(dropout_rate)(tabular_dense)
    tabular_dense = tf.keras.layers.Dense(32, activation="relu")(tabular_dense)
    tabular_dense = tf.keras.layers.Dropout(dropout_rate)(tabular_dense)
    tabular_dense = tf.keras.layers.Dense(32, activation="relu")(tabular_dense)
    tabular_dense = tf.keras.layers.Dropout(dropout_rate)(tabular_dense)
    tabular_dense = tf.keras.layers.Dense(32, activation="relu")(tabular_dense)
    # --- Concatenate ---
    concat_layer = tf.keras.layers.Concatenate()([flatten, tabular_dense])
    concat_layer = tf.keras.layers.Dropout(dropout_rate)(concat_layer)
    concat_dense = tf.keras.layers.Dense(32, activation="relu")(concat_layer)
    # Output
    output_module = tf.keras.layers.Dense(1, activation="sigmoid")(concat_dense)
    # Build module
    model = tf.keras.Model(
        inputs=[input_module, model_input_tabular],
        outputs=output_module,
        name=model_name,
    )
    # Metrics for binary classification
    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.TruePositives(),
        tf.keras.metrics.TrueNegatives(),
        tf.keras.metrics.FalsePositives(),
        tf.keras.metrics.FalseNegatives(),
    ]
    # Compile model using BinaryFocalCrossentropy
    model.compile(
        loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=metrics,
    )
    return model


# Add metrics to saved logger files
def add_metrics(hist_filelocation: str, saved_name: str) -> None:
    """Adding f1 score and specificity to CSVLogger files since they aren't available in TensorFlow.

    Args:
        hist_filelocation (str): File location of CSVLogger file.
        saved_name (str): Newfile name for csv with added metrics.
    """
    # Load data from csv and create pandas df
    training_history = pd.read_csv(hist_filelocation, sep=",")
    # Remove any "_" + number(s) at the end of the column names
    training_history.columns = [
        re.sub("_\d+", "", col) for col in training_history.columns
    ]
    # f1_score
    train_f1 = 2 * np.dot(
        np.array(training_history["precision"]), np.array(training_history["recall"])
    )
    train_f1 /= np.array(training_history["precision"]) + np.array(
        training_history["recall"]
    )
    val_f1 = 2 * np.dot(
        np.array(training_history["val_precision"]),
        np.array(training_history["val_recall"]),
    )
    val_f1 /= np.array(training_history["val_precision"]) + np.array(
        training_history["val_recall"]
    )
    training_history["f1"] = train_f1
    training_history["val_f1"] = val_f1
    # specificity
    train_specificity = np.array(training_history["true_negatives"])
    train_specificity /= np.array(training_history["true_negatives"]) + np.array(
        training_history["false_positives"]
    )
    val_specificity = np.array(training_history["val_true_negatives"])
    val_specificity /= np.array(training_history["val_true_negatives"]) + np.array(
        training_history["val_false_positives"]
    )
    training_history["specificity"] = train_specificity
    training_history["val_specificity"] = val_specificity
    # Save file
    training_history.to_csv(saved_name)


# Plotly
def save_figure(
    hist_filelocation: str,
    column1: str,
    column1_name: str,
    column2: str,
    column2_name: str,
    img_width: int,
    img_height: int,
    img_name: str,
) -> None:
    """Function to save images derived from saved CSVLogger files.

    Args:
        hist_filelocation (str): Location of logging file.
        column1 (str): Metric 1 for plotting.
        column1_name (str): Name for metric 1.
        column2 (str): Metric 2 for plotting.
        column2_name (str): Name for metric 2.
        img_width (int): Width of image for saving.
        img_height (int): Width of height for saving.
        img_name (str): Name of image to be saved.
    """
    # Load data from csv and create pandas df
    training_history = pd.read_csv(hist_filelocation, sep=",")
    epoch_array = np.array(training_history["epoch"])
    # Create image
    fig = go.Figure()
    if column1 != None:
        fig.add_trace(
            go.Scatter(
                x=epoch_array,
                y=training_history[column1],
                mode="lines",
                line=dict(color="#00235B"),
                name=f"Training: {column1_name}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epoch_array,
                y=training_history[f"val_{column1}"],
                mode="lines",
                line=dict(color="#1E63A4"),
                name=f"Validation: {column1_name}",
            )
        )
    if column2 != None:
        fig.add_trace(
            go.Scatter(
                x=epoch_array,
                y=training_history[column2],
                mode="lines",
                line=dict(color="#F05225"),
                name=f"Training: {column2_name}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=epoch_array,
                y=training_history[f"val_{column2}"],
                mode="lines",
                line=dict(color="#F1900A"),
                name=f"Validation: {column2_name}",
            )
        )

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title=None,
        title=None,
        width=img_width,
        height=img_height,
    )
    fig.write_image(f"images/{img_name}.png")


# Add Gaussian noise to images
def add_noise(imgs: np.ndarray) -> np.ndarray:
    """Add Gaussian noise with a standard deviation of 0.1 to a set of images.

    Args:
        imgs (np.ndarray): Images to add noise to.

    Returns:
        np.ndarray: Noisy images
    """
    # Restrict images to [0, 1] as before
    noisy_imgs = np.clip(imgs + np.random.normal(scale=0.1, size=imgs.shape), 0.0, 1.0)

    return noisy_imgs


def lr_scheduler(epoch):
    """Learning rate scheduler with a ramp up period.

    Source: https://wandb.ai/wandb_fc/tips/reports/How-to-Use-a-Learning-Rate-Scheduler-in-Keras--VmlldzoyMjU2MTI3

    Args:
        epoch (int): Epoch number.

    Returns:
        float: learning rate.
    """
    lr_start = 0.0005  # Initial learning rate
    lr_max = 0.0006  # Maximum learning rate during training
    lr_min = 5e-7  # Minimum learning rate
    lr_ramp_ep = 3  # The number of epochs for the learning rate to ramp up from lr_start to lr_max
    lr_sus_ep = (
        0  # The number of epochs during which the learning rate stays constant at
    )
    # lr_max after the ramp-up phase.
    lr_decay = 0.4  # The decay factor applied to the learning rate after the ramp-up and suspension phases.

    if epoch < lr_ramp_ep:
        lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

    elif epoch < lr_ramp_ep + lr_sus_ep:
        lr = lr_max

    else:
        lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min

    return lr
