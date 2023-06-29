import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_addons as tfa

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
    augmentation_module = data_augmentation(input_module)
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
