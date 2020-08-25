import tensorflow as tf
import tensorflow.keras.layers as KL


def vgg(
    dec_conv_num=4,
    dec_conv_conseq=2,
    dec_conv_filters=32,
    dec_conv_filter_size=(3, 3),
    dec_conv_activation="relu",
    dec_pool_size=2,
    dec_dense_num=2,
    dec_dense_width=128,
    dec_dense_activation="sigmoid",
    img_size=(224, 224),
    dropout=0.3,
    num_classes=2,
):

    input_explanation_layer = KL.Input(
        tuple(list(img_size) + [1]), name="classifier-explanation-input"
    )

    input_image_layer = KL.Input(
        tuple(list(img_size) + [3]), name="classifier-image-input"
    )

    last_expl = input_explanation_layer
    last_dec = input_image_layer

    # Convolutional section
    for conv_level in range(dec_conv_num):
        nfilters = dec_conv_filters * 2 ** conv_level

        # Convolutional layers
        for c in range(dec_conv_conseq):
            cfs = dec_conv_filter_size

            conv_dec = KL.Conv2D(
                nfilters,
                cfs,
                padding="same",
                activation=dec_conv_activation,
                name="classifier-image-conv%d-%d" % (conv_level, c),
            )
            last_dec = conv_dec(last_dec)

        last_dec = KL.Multiply()(
            [KL.Concatenate()([last_expl] * nfilters), last_dec]  # last_expl,
        )

        # Pooling layer
        last_expl = KL.AveragePooling2D(
            pool_size=(dec_pool_size, dec_pool_size),
            name="classifier-explanation-pool%d" % conv_level,
        )(last_expl)
        last_dec = KL.MaxPool2D(
            pool_size=(dec_pool_size, dec_pool_size),
            name="classifier-image-pool%d" % conv_level,
        )(last_dec)

    last_dec = KL.GlobalMaxPool2D()(last_dec)

    for _ in range(dec_dense_num):
        last_dec = KL.Dense(dec_dense_width, activation=dec_dense_activation)(last_dec)

        if dropout is not None and dropout > 0:
            last_dec = KL.Dropout(rate=dropout, seed=42)(last_dec)

    last_dec = KL.Dense(num_classes, activation="softmax")(last_dec)

    return tf.keras.Model(
        inputs=[input_explanation_layer, input_image_layer],
        outputs=[last_dec],
        name="classifier",
    )
