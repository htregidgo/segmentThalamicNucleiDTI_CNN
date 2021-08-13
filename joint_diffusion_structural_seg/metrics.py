import keras.backend as K
import tensorflow as tf

class WL2Loss(object):

    def __init__(self, target_value, n_labels, background_weight=1e-4, **kwargs):
        self.target_value = target_value
        self.n_labels = n_labels
        self.background_weight = background_weight

    def loss(self, gt, pred):
        weights = tf.expand_dims(1 - gt[..., 0] + self.background_weight, -1)
        loss = K.sum(weights * K.square(pred - self.target_value * (2 * gt - 1))) / (K.sum(weights) * self.n_labels)
        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss

class DiceLoss(object):

    def __init__(self, **kwargs):
        pass

    def loss(self, y, x):
        x = K.clip(x / tf.math.reduce_sum(x, axis=-1, keepdims=True), 0, 1)
        y = K.clip(y / tf.math.reduce_sum(y, axis=-1, keepdims=True), 0, 1)
        # compute dice loss for each label
        top = tf.math.reduce_sum(2 * x * y, axis=list(range(1, 4)))
        bottom = tf.math.square(x) + tf.math.square(y) + tf.keras.backend.epsilon()
        bottom = tf.math.reduce_sum(bottom, axis=list(range(1, 4)))
        dice = top / bottom
        loss = 1 - dice
        tf.debugging.check_numerics(loss, 'Loss not finite')
        return loss