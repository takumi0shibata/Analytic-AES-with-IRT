from tensorflow.keras import layers, regularizers
import tensorflow.keras.backend as K
import tensorflow as tf


class MGPCM(layers.Layer):
    def __init__(self, num_item, latent_dim, num_category, inf_mask=None, beta_mask=None, beta_penalty=1e-6, **kwargs):
        super(MGPCM, self).__init__(**kwargs)
        self.num_category = num_category
        self.latent_dim = latent_dim
        self.inf_mask = inf_mask
        self.beta_mask = beta_mask

        self.alpha = self.add_weight(
            shape=(num_item, latent_dim),
            initializer='random_normal',
            constraint=tf.keras.constraints.non_neg(),
            trainable=True
        )
        self.beta = self.add_weight(
            shape=(num_item, num_category-1),
            initializer='random_normal',
            regularizer=regularizers.l2(beta_penalty),
            trainable=True
        )

    def call(self, theta, mask=True):
        beta = tf.pad(self.beta, [[0, 0], [1, 0]])
        beta = K.cumsum(beta, axis=-1)
        if mask:
            beta = beta * self.beta_mask

        x = K.dot(theta, K.transpose(self.alpha))
        x = K.expand_dims(x, axis=-1)
        x = K.repeat_elements(x, self.num_category, axis=-1)
        x = K.cumsum(x, axis=-1)
        x = x + beta
        if mask:
            x = x + self.inf_mask
        x = K.softmax(x)

        return x
