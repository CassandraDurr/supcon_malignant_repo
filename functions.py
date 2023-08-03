import os
import re

import numpy as np
import matplotlib.pylot as plt
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from kerasgen.balanced_image_dataset import balanced_image_dataset_from_directory
import tensorflow_addons as tfa
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


# ViT functions
class SoftAttention(tf.keras.layers.Layer):
    def __init__(self, ch, m, concat_with_x=False, aggregate=False, **kwargs):
        self.channels = int(ch)
        self.multiheads = m
        self.aggregate_channels = aggregate
        self.concat_input_with_scaled = concat_with_x

        super(SoftAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.i_shape = input_shape

        kernel_shape_conv3d = (self.channels, 3, 3) + (1, self.multiheads)  # DHWC

        self.out_attention_maps_shape = (
            input_shape[0:1] + (self.multiheads,) + input_shape[1:-1]
        )

        if self.aggregate_channels == False:
            self.out_features_shape = input_shape[:-1] + (
                input_shape[-1] + (input_shape[-1] * self.multiheads),
            )
        else:
            if self.concat_input_with_scaled:
                self.out_features_shape = input_shape[:-1] + (input_shape[-1] * 2,)
            else:
                self.out_features_shape = input_shape

        self.kernel_conv3d = self.add_weight(
            shape=kernel_shape_conv3d, initializer="he_uniform", name="kernel_conv3d"
        )
        self.bias_conv3d = self.add_weight(
            shape=(self.multiheads,), initializer="zeros", name="bias_conv3d"
        )

        super(SoftAttention, self).build(input_shape)

    def call(self, x):
        exp_x = tf.keras.backend.expand_dims(x, axis=-1)

        c3d = tf.keras.backend.conv3d(
            exp_x,
            kernel=self.kernel_conv3d,
            strides=(1, 1, self.i_shape[-1]),
            padding="same",
            data_format="channels_last",
        )
        conv3d = tf.keras.backend.bias_add(c3d, self.bias_conv3d)
        conv3d = tf.keras.layers.Activation("relu")(conv3d)
        conv3d = tf.keras.backend.permute_dimensions(conv3d, pattern=(0, 4, 1, 2, 3))
        conv3d = tf.keras.backend.squeeze(conv3d, axis=-1)
        conv3d = tf.keras.backend.reshape(
            conv3d, shape=(-1, self.multiheads, self.i_shape[1] * self.i_shape[2])
        )

        softmax_alpha = tf.keras.backend.softmax(conv3d, axis=-1)
        softmax_alpha = tf.keras.layers.Reshape(
            target_shape=(self.multiheads, self.i_shape[1], self.i_shape[2])
        )(softmax_alpha)

        if self.aggregate_channels == False:
            exp_softmax_alpha = tf.keras.backend.expand_dims(softmax_alpha, axis=-1)
            exp_softmax_alpha = tf.keras.backend.permute_dimensions(
                exp_softmax_alpha, pattern=(0, 2, 3, 1, 4)
            )

            x_exp = tf.keras.backend.expand_dims(x, axis=-2)

            u = tf.keras.layers.Multiply()([exp_softmax_alpha, x_exp])
            u = tf.keras.layers.Reshape(
                target_shape=(
                    self.i_shape[1],
                    self.i_shape[2],
                    u.shape[-1] * u.shape[-2],
                )
            )(u)

        else:
            exp_softmax_alpha = tf.keras.backend.permute_dimensions(
                softmax_alpha, pattern=(0, 2, 3, 1)
            )
            exp_softmax_alpha = tf.keras.backend.sum(exp_softmax_alpha, axis=-1)
            exp_softmax_alpha = tf.keras.backend.expand_dims(exp_softmax_alpha, axis=-1)

            u = tf.keras.layers.Multiply()([exp_softmax_alpha, x])

        if self.concat_input_with_scaled:
            o = tf.keras.layers.Concatenate(axis=-1)([u, x])
        else:
            o = u

        return [o, softmax_alpha]

    def compute_output_shape(self, input_shape):
        return [self.out_features_shape, self.out_attention_maps_shape]

    def get_config(self):
        return super(SoftAttention, self).get_config()


# https://keras.io/examples/vision/image_classification_with_vision_transformer/
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


# ViT encoder
def create_vit_encoder(
    data_augmentation: tf.keras.Sequential,
    patch_size: int,
    num_patches: int,
    projection_dim: int,
    transformer_layers: int,
    num_heads: int,
    transformer_units: list,
    input_shape: tuple,
    encoder_name: str,
    representation_units: int,
    load_weights: bool,
    weight_location: str,
) -> tf.keras.Model:
    # Create model with the chosen encoder
    input_module = tf.keras.Input(shape=input_shape)
    rescale = tf.keras.layers.Rescaling(scale=1.0 / 255, offset=0.0)(input_module)
    augmentation_module = data_augmentation(rescale)
    # Create encoder
    vit_encoder = create_vit_encoder_module(
        patch_size=patch_size,
        num_patches=num_patches,
        projection_dim=projection_dim,
        transformer_layers=transformer_layers,
        num_heads=num_heads,
        transformer_units=transformer_units,
        input_shape=input_shape,
        encoder_name="ViT_encoder_module",
        representation_units=representation_units,
    )
    if load_weights:
        vit_encoder.load_weights(weight_location)

    output_module = vit_encoder(augmentation_module)

    model = tf.keras.Model(
        inputs=input_module, outputs=output_module, name=encoder_name
    )

    return model


# Create ViT encoder module without augmentation
def create_vit_encoder_module(
    patch_size: int,
    num_patches: int,
    projection_dim: int,
    transformer_layers: int,
    num_heads: int,
    transformer_units: list,
    input_shape: tuple,
    encoder_name: str,
    representation_units: int,
) -> tf.keras.Model:
    # Create model with the chosen encoder
    input_module = tf.keras.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(input_module)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)

    # Add MLP.
    output_module = tf.keras.layers.Dense(representation_units, activation=tf.nn.gelu)(
        representation
    )

    model = tf.keras.Model(
        inputs=input_module, outputs=output_module, name=encoder_name
    )

    return model


# Create ViT decoder
def create_decoder():
    # Start from encoded input shape
    decoder_input = tf.keras.layers.Input(shape=(2048,))

    # Reshape it to something spatial, assuming the encoder used convolutions
    x = tf.keras.layers.Reshape((8, 8, 32))(decoder_input)

    # Upsampling to the original size
    # 8x8x32 -> 16x16x128
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # 16x16x128 -> 32x32x64
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # 32x32x64 -> 64x64x32
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # 64x64x32 -> 128x128x16
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # 128x128x16 -> 256x256x16
    x = tf.keras.layers.UpSampling2D()(x)

    # 256x256x16 -> 224x224x3
    # Here we use a Conv2D with kernel size 33 to reduce the spatial dimensions
    x = tf.keras.layers.Conv2D(3, (33, 33), activation="sigmoid", padding="valid")(x)

    # Build the model
    decoder = tf.keras.Model(decoder_input, x)

    return decoder


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
    rescale = tf.keras.layers.Rescaling(scale=1.0 / 255, offset=0.0)(input_module)
    augmentation_module = data_augmentation(rescale)
    output_module = encoder_module(augmentation_module)
    output_module = tf.keras.layers.GlobalAveragePooling2D()(output_module)
    model = tf.keras.Model(
        inputs=input_module, outputs=output_module, name=encoder_name
    )
    return model


# Create data-augmentation module
def create_data_augmentation_module(
    RandomRotationAmount: float = 0.5,
) -> tf.keras.Sequential:
    """This function creates a data augmentation module for TensorFlow models.

    Args:
        RandomRotationAmount (float, optional): Argument for tf.keras.layers.RandomRotation. Defaults to 0.5.

    Returns:
        tf.keras.Sequential: Data augmentation module
    """
    data_augmentation = tf.keras.Sequential(
        [
            # tf.keras.layers.Normalization(),
            tf.keras.layers.GaussianNoise(stddev=0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomFlip("vertical"),
            tf.keras.layers.RandomRotation(RandomRotationAmount),
            tf.keras.layers.RandomZoom(height_factor=(0.0, 0.1)),
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
    # Compile model using Binary Crossentropy
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
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
    # Compile model using Binary Crossentropy
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
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


# Load data generators
def load_generators(
    trainDataDir: str,
    validDataDir: str,
    testDataDir: str,
    image_width: int,
    num_images_per_class: int,
):
    # Create balanced traing and validation datasets
    batch_size = int(2 * num_images_per_class)
    train_ds = balanced_image_dataset_from_directory(
        trainDataDir,
        num_classes_per_batch=2,
        num_images_per_class=num_images_per_class,
        image_size=(image_width, image_width),
        seed=980801,
        safe_triplet=True,
    )

    val_ds = balanced_image_dataset_from_directory(
        validDataDir,
        num_classes_per_batch=2,
        num_images_per_class=num_images_per_class,
        image_size=(image_width, image_width),
        seed=980801,
        safe_triplet=True,
    )

    # Create an ImageDataGenerator for testing data (don't need balanced test sets)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # Create a test generator using the test directory
    test_ds = test_datagen.flow_from_directory(
        testDataDir,
        target_size=(image_width, image_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    # To select the optimal threshold we will also create a data generator
    # for validation that is unbalanced
    valid_datagen_unbalanced = tf.keras.preprocessing.image.ImageDataGenerator()
    valid_ds_unbalanced = valid_datagen_unbalanced.flow_from_directory(
        validDataDir,
        target_size=(image_width, image_width),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    return train_ds, val_ds, valid_ds_unbalanced, test_ds


# Set threshold for classification based on roc
def find_optimal_threshold(classifier, valid_dataset, test_dataset):
    beta_val = 2.0

    # Get predicted probabilities on the validation
    train_probs = classifier.predict(valid_dataset)

    # Calculate ROC curve on the validation dataset
    # fpr, tpr, thresholds = roc_curve(valid_dataset.labels, train_probs)
    # # Find the optimal threshold based on ROC curve
    # optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    # Choose threshold based on f1 score
    thresholds = [0.35, 0.4, 0.425, 0.45, 0.475, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    f1_scores = []
    for threshold in thresholds:
        # Classify instances as positive or negative based on the threshold
        valid_predictions = (train_probs > threshold).astype(int)
        accuracy = fbeta_score(valid_dataset.labels, valid_predictions, beta=beta_val)
        f1_scores.append(accuracy)

    # Find the threshold that gives the highest accuracy
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    # Get predicted probabilities on the test dataset
    test_probs = classifier.predict(test_dataset)

    # Classify instances as positive or negative based on the optimal threshold
    test_predictions = (test_probs > optimal_threshold).astype(int)

    # Evaluate the model's performance on the test dataset with the new threshold
    test_auc_score = roc_auc_score(test_dataset.labels, test_probs)
    test_accuracy = accuracy_score(test_dataset.labels, test_predictions)
    test_precision = precision_score(test_dataset.labels, test_predictions)
    test_recall = recall_score(test_dataset.labels, test_predictions)
    # Specificity = recall for negative class (label 0)
    test_specificity = recall_score(test_dataset.labels, test_predictions, pos_label=0)
    test_f1 = fbeta_score(test_dataset.labels, test_predictions, beta=beta_val)
    test_metrics = {
        "auc": test_auc_score,
        "accuracy": test_accuracy,
        "precision": test_precision,
        "recall": test_recall,
        "specificity": test_specificity,
        "f1": test_f1,
    }

    return test_predictions, optimal_threshold, test_metrics


# Semi-balanced
def custom_data_generator(
    data_dir: str, batch_size: int, class_0_ratio: float = 0.8, image_width: int = 224
):
    class_0_dir = os.path.join(data_dir, "0")
    class_1_dir = os.path.join(data_dir, "1")
    class_0_files = os.listdir(class_0_dir)
    class_1_files = os.listdir(class_1_dir)

    while True:
        batch_class_0 = np.random.choice(
            class_0_files, size=int(batch_size * class_0_ratio), replace=False
        )
        batch_class_1 = np.random.choice(
            class_1_files, size=int(batch_size * (1 - class_0_ratio)), replace=False
        )

        batch_files = np.concatenate([batch_class_0, batch_class_1])
        np.random.shuffle(batch_files)

        batch_images = []
        batch_labels = []
        for file in batch_files:
            label = 1 - int(
                file in batch_class_0
            )  # Assign 0 or 1 based on class folder
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(data_dir, str(label), file),
                target_size=(image_width, image_width),
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            batch_images.append(img_array)
            batch_labels.append(label)

        yield np.concatenate(batch_images), np.array(batch_labels)


# Build ResNet autoencoder
def build_resnet_autoencoder(
    encoder_module: tf.keras.Model, input_shape: tuple[int], input_layer: tf.keras.layers.Input
) -> tf.keras.Model:
    
    # The output of the encoder
    output_encoder = encoder_module(input_layer)

    encoder_model = tf.keras.Model(inputs=input_layer, outputs=output_encoder)
    print("Encoder summary")
    encoder_model.summary()
    print("\n")

    # Define decoder
    decoder_input = tf.keras.layers.Input(shape=(7, 7, 2048))
    # Upsample the features using transpose convolutional layers
    x = tf.keras.layers.Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding="same")(
        decoder_input
    )
    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)
    decoder_output = tf.keras.layers.Conv2D(
        3, (3, 3), activation="sigmoid", padding="same"
    )(x)
    decoder_model = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)
    print("\nDecoder summary")
    decoder_model.summary()
    print("\n")

    # Build the autoencoder
    ae_inputs = tf.keras.layers.Input(shape=input_shape)
    # Noise
    add_noise = tf.keras.layers.GaussianNoise(stddev=0.1)(ae_inputs)
    encoder_out = encoder_model(add_noise)
    decoder_out = decoder_model(encoder_out)
    autoencoder_model = tf.keras.Model(inputs=ae_inputs, outputs=decoder_out)

    # Print the summary of the autoencoder model
    print("\nAutoencoder summary")
    autoencoder_model.summary()

    return autoencoder_model


# # Build InceptionV3 pretext autoencoder
def build_inception_autoencoder(
    encoder_module: tf.keras.Model, input_shape: tuple[int], input_layer: tf.keras.layers.Input
) -> tf.keras.Model:
    
    # Output of encoder
    output_encoder = encoder_module(input_layer)

    encoder_model = tf.keras.Model(inputs=input_layer, outputs=output_encoder)
    print("Encoder summary")
    encoder_model.summary()
    print("\n")

    # Define decoder
    decoder_input = tf.keras.layers.Input(shape=(5, 5, 2048))
    # Upsample the features using transpose convolutional layers
    x = tf.keras.layers.Conv2DTranspose(1024, (3, 3), strides=(2, 2), padding="same")(
        decoder_input
    )
    x = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(x)

    # Output layer for deconvolution
    x = tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same")(x)
    decoder_output = tf.image.resize(x, (224, 224), method="bicubic")
    decoder_output = tf.clip_by_value(
        decoder_output, clip_value_min=0.0, clip_value_max=1.0
    )
    decoder_model = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)
    print("\nDecoder summary")
    decoder_model.summary()
    print("\n")

    # Build the autoencoder
    ae_inputs = tf.keras.layers.Input(shape=input_shape)
    # Noise
    add_noise = tf.keras.layers.GaussianNoise(stddev=0.1)(ae_inputs)
    encoder_out = encoder_model(add_noise)
    decoder_out = decoder_model(encoder_out)
    autoencoder_model = tf.keras.Model(inputs=ae_inputs, outputs=decoder_out)

    # Print the summary of the autoencoder model
    print("\nAutoencoder summary")
    autoencoder_model.summary()

    return autoencoder_model


# Build ViT pretext denoising autoencoder
def build_vit_autoencoder(
    encoder_module: tf.keras.Model, input_shape: tuple[int], input_layer: tf.keras.layers.Input
) -> tf.keras.Model:
    
    output_encoder = encoder_module(input_layer)

    encoder_model = tf.keras.Model(inputs=input_layer, outputs=output_encoder)
    print("Encoder summary")
    encoder_model.summary()
    print("\n")

    # Define decoder
    decoder_model = create_decoder()
    print("\nDecoder summary")
    decoder_model.summary()
    print("\n")

    # Build the autoencoder
    ae_inputs = tf.keras.layers.Input(shape=input_shape)
    # Noise
    add_noise = tf.keras.layers.GaussianNoise(stddev=0.1)(ae_inputs)
    encoder_out = encoder_model(add_noise)
    decoder_out = decoder_model(encoder_out)
    autoencoder_model = tf.keras.Model(inputs=ae_inputs, outputs=decoder_out)

    # Print the summary of the autoencoder model
    print("\nAutoencoder summary")
    autoencoder_model.summary()

    return autoencoder_model


# Visualise pretext-training images
def visualise_pretext(
    img_paths: list[str], images: list[np.ndarray], denoised_imgs: np.ndarray
) -> None:
    num_images = len(img_paths)
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 2 * num_images))

    for i in range(num_images):
        # Display original noisy image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Noisy Image")
        axes[i, 0].axis("off")

        # Display denoised image
        axes[i, 1].imshow(denoised_imgs[i])
        axes[i, 1].set_title("Denoised Image")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
