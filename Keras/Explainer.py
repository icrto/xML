import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras.initializers import Constant


def explainer(
    exp_conv_num=3,
    exp_conv_conseq=2,
    exp_conv_filters=32,
    exp_conv_filter_size=(3, 3),
    exp_conv_activation="relu",
    exp_pool_size=2,
    img_size=(224, 224),
    init_bias=10.0,
):
    """explainer explainer architecture constructor

    Keyword Arguments:
        exp_conv_num {int} --  number of convolutional stages (default: {3})
        exp_conv_conseq {int} -- number of consecutive convolutional layers (default: {2})
        exp_conv_filters {int} -- number of initial filters (default: {32})
        exp_conv_filter_size {tuple} -- kernel size for convolutional layers (default: {(3, 3)})
        exp_conv_activation {str} -- activation function for convolutional layers (default: {"relu"})
        exp_pool_size {int} --  pooling factor (default: {2})
        img_size {tuple} -- image dimensions (default: {(224, 224)})
        init_bias {float} -- initial bias for batch normalisation layer (default: {10.0})

    Returns:
        tf.keras.Model -- explainer model
    """

    input_layer = KL.Input(tuple(list(img_size) + [3]), name="explainer-input")
    last = input_layer

    # Convolutional section
    last_conv_per_level = []
    sizes = []
    for conv_level in range(exp_conv_num):
        nfilters = exp_conv_filters * 2 ** conv_level
        sizes.append(nfilters)

        # Convolutional layers
        for c in range(exp_conv_conseq):
            cfs = exp_conv_filter_size

            last = KL.Conv2D(
                nfilters,
                cfs,
                activation=exp_conv_activation,
                padding="same",
                name="explainer-conv%d-%d" % (conv_level, c),
            )(last)

        last_conv_per_level.append(last)

        # Pooling layer
        if conv_level != exp_conv_num:
            last = KL.MaxPool2D(
                pool_size=(exp_pool_size, exp_pool_size),
                name="explainer-pool%d" % conv_level,
            )(last)

    # Deconvolutional section
    for conv_level in range(exp_conv_num)[::-1]:
        cc = KL.Add(
            name="explainer-add%d" % conv_level
        )  # u-net connection from encoder to decoder

        last = cc(
            [
                KL.Conv2DTranspose(
                    sizes[conv_level],
                    exp_pool_size,
                    strides=(exp_pool_size, exp_pool_size),
                    name="explainer-transpose%d-%d" % (conv_level, 0),
                )(last),
                last_conv_per_level[conv_level],
            ]
        )

        for c in range(exp_conv_conseq):
            cfs = exp_conv_filter_size

            last = KL.Conv2D(
                sizes[conv_level],
                cfs,
                activation=exp_conv_activation,
                padding="same",
                name="explainer-deconv%d-%d" % (conv_level, c),
            )(last)

    last = KL.Conv2D(1, (1, 1), activation="linear", name="explainer-conv-output",)(
        last
    )
    # The beta_initializer parameter initialises the batch norm layer's bias to a pre-defined value,
    # guaranteeing that the initially produced explanations are white images (filled with ones) --> see paper for details.
    last = KL.BatchNormalization(beta_initializer=Constant(value=init_bias))(last)
    last = KL.Activation("tanh")(last)
    last = KL.Activation("relu")(last)

    out = last

    return tf.keras.Model(inputs=[input_layer], outputs=[out], name="explainer")

