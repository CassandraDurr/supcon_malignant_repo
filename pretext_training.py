"""Pre-text training task, image de-noising, for supervised contrastive learning. Dataset used is different to training dataset."""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from functions import lr_scheduler

# Output shape of resnet50v2 = (None, 7, 7, 2048)
# Output shape of inceptionv3 = (None, 5, 5, 2048)

input_shape = (224, 224, 3)
batch_size = 10  # 32
lr_start = 0.0005

encoder_type = "InceptionV3" # "ResNet50V2"
input_layer = tf.keras.layers.Input(shape=input_shape)
if encoder_type == "ResNet50V2":
    print(f"\n{encoder_type}\n")
    encoder_module = tf.keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape
    )

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

else:
    print(f"\n{encoder_type}\n")
    encoder_module = tf.keras.applications.InceptionV3(
        include_top=False, weights=None, input_shape=input_shape
    )

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
    decoder_output = tf.image.resize(x, (224, 224), method = "bicubic")
    decoder_output = tf.clip_by_value(decoder_output, clip_value_min=0.0, clip_value_max=1.0)
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

# Create data generator
# Dataset = Stanford DDI (Diverse Dermatology Images)
# https://ddi-dataset.github.io/
# D:\Downloads\ddidiversedermatologyimages
folder_imgs = "D:/Downloads/pretext_task/"
image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2, rescale=1.0 / 255, horizontal_flip=True,
    vertical_flip=True
)
image_generator = image_data_generator.flow_from_directory(
    folder_imgs,
    color_mode="rgb",
    shuffle=True,
    seed=20221015,
    subset="training",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="input",
)
val_image_generator = image_data_generator.flow_from_directory(
    folder_imgs,
    color_mode="rgb",
    shuffle=True,
    seed=20221015,
    subset="validation",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="input",
)

# Print the summary of the autoencoder model
autoencoder_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_start),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[
        tf.keras.metrics.BinaryCrossentropy(name="binary_crossentropy"),
        tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
    ],
)

# Fit the autoencoder to the data
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
csv_callback = tf.keras.callbacks.CSVLogger(
    f"CSVLogger/model_autoencoder_{encoder_type}_MSE.csv"
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
history = autoencoder_model.fit(
    image_generator,
    validation_data=val_image_generator,
    epochs=100,
    callbacks=[es_callback, csv_callback, lr_callback],
)

# Try to save the weights from the encoder
encoder_model.save_weights(f"saved_models/encoder_weights_{encoder_type}_MSE.h5")
# encoder_model.load_weights('encoder_weights.h5')

# Plots
# for metric in ["loss", "mean_squared_error", "root_mean_squared_error"]:
for metric in ["loss", "binary_crossentropy", "root_mean_squared_error"]:
    plt.figure(figsize=(9, 6))
    plt.plot(history.history[metric], label="Training")
    plt.plot(history.history[f"val_{metric}"], label="Validation")
    plt.ylabel(metric)
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"images/train_{metric}_autoencoder_{encoder_type}_MSE.png")
    plt.show()

# Visually assess the model
root_path = "D:/Downloads/pretext_task/ddidiversedermatologyimages/"
img_paths = [root_path + "000383.png", root_path + "000581.png", root_path + "000573.png", root_path + "000206.png", root_path + "000028.png"]
print(img_paths)

images = []
for path in img_paths:
    image = tf.keras.utils.load_img(path, target_size=(224, 224))  # Set the desired target size
    image_array = tf.keras.utils.img_to_array(image)
    images.append(image_array)

# Rescale and reshape
images = np.array(images)
images /= 255.0
images = images.reshape(-1, 224, 224, 3)

denoised_imgs = autoencoder_model.predict(images)

# Visualize the denoised images alongside the original noisy images
num_images = len(img_paths)
fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 2*num_images))

for i in range(num_images):
    # Display original noisy image
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title('Noisy Image')
    axes[i, 0].axis('off')

    # Display denoised image
    axes[i, 1].imshow(denoised_imgs[i])
    axes[i, 1].set_title('Denoised Image')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()