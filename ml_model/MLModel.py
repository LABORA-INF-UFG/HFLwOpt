import tensorflow as tf


class Model:
    @staticmethod
    def create_model(cm):
        if cm.model_type == "MLP":
            return Model.create_model_mlp(cm.shape)
        else:
            return Model.create_model_cnn(cm.shape)


    @staticmethod
    def create_model_mlp(shape):
        model = tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=(shape[0]*shape[1]),),
                tf.keras.layers.Dense(units=192),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(10, activation="softmax")
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return model

    @staticmethod
    def create_model_cnn(shape):

        model = tf.keras.models.Sequential([

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=192),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy']
                      )

        return model
