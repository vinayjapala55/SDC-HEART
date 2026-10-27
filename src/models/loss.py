"""
Masked and Weighted Multi-Label Binary Cross-Entropy Loss for RosaNet.

Supports:
- Class-specific positive weights for imbalanced multilabel classification:
  w = [N, AS, AR, MR, MS]
- Unknown / unannotated label masking:
  Labels equal to -1.0 (e.g., patient_022 MVP unknown targets) are masked out
  with 0 loss contribution and 0 gradient penalty.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="RosaNet")
class WeightedMultiLabelBCE(tf.keras.losses.Loss):
    """
    Computes positive-class-weighted Multi-Label Binary Cross-Entropy Loss with unknown-label masking:
      L = - (1 / sum(valid_mask)) * sum_{c, valid} [ w_c * y_c * log(p_c + eps) + (1 - y_c) * log(1 - p_c + eps) ]
    Entries where y_true == -1.0 are masked and ignored.
    """
    def __init__(self, pos_weights=None, epsilon=1e-7, name="weighted_multilabel_bce", **kwargs):
        super(WeightedMultiLabelBCE, self).__init__(name=name, **kwargs)
        if pos_weights is None:
            pos_weights = [2.62, 4.85, 3.22, 1.92, 1.92]
        self.pos_weights = tf.constant(pos_weights, dtype=tf.float32)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1.0 - self.epsilon)

        # Mask out unknown labels (-1.0)
        valid_mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        y_true_clean = tf.maximum(y_true, 0.0)

        pos_term = self.pos_weights * y_true_clean * tf.math.log(y_pred)
        neg_term = (1.0 - y_true_clean) * tf.math.log(1.0 - y_pred)
        unmasked_loss = - (pos_term + neg_term)
        masked_loss = unmasked_loss * valid_mask

        valid_count = tf.reduce_sum(valid_mask)
        return tf.reduce_sum(masked_loss) / tf.maximum(valid_count, 1.0)

    def get_config(self):
        config = super(WeightedMultiLabelBCE, self).get_config()
        config.update({
            'pos_weights': self.pos_weights.numpy().tolist(),
            'epsilon': self.epsilon
        })
        return config


# Alias for explicit clarity
MaskedWeightedMultiLabelBCE = WeightedMultiLabelBCE
