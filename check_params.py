import tensorflow as tf

name = "saved_models\supcon_encoder_ResNet50V2_incl_tabular"

model = tf.keras.models.load_model(name)
model.summary()