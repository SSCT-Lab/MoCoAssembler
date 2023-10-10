"""Define the model."""

import tensorflow as tf

def build_model(params, input_shape = (227, 227, 3), classes = 10):
    """Compute logits of the model (output distribution)
    Args:
        params: (Params) hyperparameters
        input_shape: (tuple) shape of a input image
        classes: (int) number of classes of image
    Returns:
        model: (tf.keras.Model) compiled model
    """
    x_input = tf.keras.Input(input_shape)
    x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3))(x_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(4096, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    # Create model
    model = tf.keras.Model(inputs = x_input, outputs = x, name='ALEXNET')
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate), 
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model