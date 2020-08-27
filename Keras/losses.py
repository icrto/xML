import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

eps = 1e-6  # additive constant to avoid divisions by 0


def unsupervised_explanation_loss(beta):
    """unsupervised_explanation_loss

    Arguments:
        beta {float} -- beta hyperparameter (weighs each unsupervised loss element)
    """

    def loss(y_true, y_pred):
        """loss [summary]

        Arguments:
            y_true {numpy array} -- batched ground-truth
            y_pred {numpy array} -- batched predictions

        Returns:
            float -- unsupervised loss
        """
        return beta * K.mean(K.abs(y_pred)) + (1 - beta) * (
            K.mean(K.abs(y_pred[:-1] - y_pred[1:]))
            + K.mean(K.abs(y_pred[:, :-1] - y_pred[:, 1:]))
        )

    return loss


def hybrid_explanation_loss(beta, gamma):
    """hybrid_explanation_loss

    Arguments:
        beta {float} -- beta hyperparameter (weighs each unsupervised loss element)
        gamma {float} -- gamma hyperparameter (weighs the weakly supervised loss element)
    """

    def hybridloss(y_true, y_pred):
        """hybridloss [summary]

        Arguments:
            y_true {numpy array} -- batched ground-truth
            y_pred {numpy array} -- batched predictions

        Returns:
            float -- hybrid loss
        """

        l1 = K.mean(K.abs(y_pred))
        tv = K.mean(K.abs(y_pred[:-1] - y_pred[1:])) + K.mean(
            K.abs(y_pred[:, :-1] - y_pred[:, 1:])
        )

        y_true = tf.cast(y_true, tf.float32)
        weakly = tf.math.abs(
            tf.math.divide(
                tf.math.reduce_sum(tf.math.multiply(1 - y_true, y_pred)),
                tf.math.reduce_sum(1 - y_true) + eps,
            )
        )
        return beta * l1 + (1 - beta) * tv + gamma * weakly

    return hybridloss
