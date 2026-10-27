"""
Attention Mechanisms for the RosaNet Heart Sound Classification Architecture.

Flow:
======================================================================
INPUT FEATURES
      ↓
ATTENTION MECHANISM (Channel & Temporal / Split-Self-Attention)
      ↓
LEARNED IMPORTANCE WEIGHTS (alpha in [0, 1])
      ↓
WEIGHTED FEATURES (Input * alpha)
======================================================================

Purpose:
Allows the network to learn which temporal acoustic segments (e.g. S1/S2 heart
sounds, systolic ejection murmurs, diastolic rumbles) and frequency-band features
contain clinically discriminative information, rather than treating all spectrogram
regions equally.

Note: Attention re-weights learned feature representations for classification
performance; formal clinical explainability is handled downstream via XAI tools.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="RosaNet")
class SplitSelfAttention(layers.Layer):
    """
    Split-Self-Attention Mechanism for RosaNet.
    
    Divides the input feature channels into two equal halves (X1, X2).
    X2 passes through a nonlinear projection to compute attention weights,
    which modulate X1. The modulated half and unmodulated half are recombined.
    
    Formula:
        X1, X2 = split(X, 2, axis=-1)
        Attention_Weights = Sigmoid(Conv(ReLU(Conv(X2))))
        Weighted_X1 = X1 * Attention_Weights
        Output = Concatenate([Weighted_X1, X2], axis=-1)
    """
    def __init__(self, channels: int, **kwargs):
        super(SplitSelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.half_channels = channels // 2
        
        self.att_conv1 = layers.Conv2D(
            filters=self.half_channels // 2 if self.half_channels >= 4 else self.half_channels,
            kernel_size=1,
            padding='same',
            use_bias=True,
            name='att_proj_reduce'
        )
        self.att_relu = layers.Activation('relu', name='att_relu')
        self.att_conv2 = layers.Conv2D(
            filters=self.half_channels,
            kernel_size=1,
            padding='same',
            use_bias=True,
            name='att_proj_expand'
        )
        self.att_sigmoid = layers.Activation('sigmoid', name='att_sigmoid')

    def call(self, inputs, training=None):
        # 1. INPUT FEATURES: (B, H, W, C)
        # Split along channel dimension into two equal partitions
        x1 = inputs[..., :self.half_channels]
        x2 = inputs[..., self.half_channels:]
        
        # 2. ATTENTION: Compute attention weights from the second partition
        attn = self.att_conv1(x2)
        attn = self.att_relu(attn)
        attn = self.att_conv2(attn)
        
        # 3. LEARNED IMPORTANCE WEIGHTS: In range [0, 1]
        importance_weights = self.att_sigmoid(attn)
        
        # 4. WEIGHTED FEATURES: Modulate the first feature partition
        weighted_x1 = layers.multiply([x1, importance_weights])
        
        # Recombine to preserve the full feature dimension
        output = layers.concatenate([weighted_x1, x2], axis=-1)
        return output

    def get_config(self):
        config = super(SplitSelfAttention, self).get_config()
        config.update({'channels': self.channels})
        return config


@tf.keras.utils.register_keras_serializable(package="RosaNet")
class CardiacSpatialTemporalAttention(layers.Layer):
    """
    Dual Temporal and Channel Attention Block for Heart Sound Spectrograms.
    
    1. Temporal Attention: Identifies specific cardiac cycle time windows
       (systole vs diastole) containing diagnostic murmurs.
    2. Channel Attention: Focuses on specific acoustic feature channels
       encoding characteristic murmur frequency bands.
       
    Flow:
        INPUT FEATURES (B, T, C)
               ↓
        ATTENTION (FC layers with reduction)
               ↓
        LEARNED IMPORTANCE WEIGHTS alpha (B, T, C) in [0, 1]
               ↓
        WEIGHTED FEATURES (Input * alpha)
    """
    def __init__(self, feature_dim: int, reduction_ratio: int = 4, **kwargs):
        super(CardiacSpatialTemporalAttention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.reduction_ratio = reduction_ratio
        
        reduced_dim = max(4, feature_dim // reduction_ratio)
        self.dense_reduce = layers.Dense(reduced_dim, activation='relu', name='att_dense_reduce')
        self.dense_expand = layers.Dense(feature_dim, activation='sigmoid', name='att_dense_expand')

    def call(self, inputs, training=None):
        # inputs shape: (B, T, C) or (B, C)
        # Compute importance weights
        weights = self.dense_reduce(inputs)
        importance_weights = self.dense_expand(weights)
        
        # Apply learned importance weights
        weighted_features = layers.multiply([inputs, importance_weights])
        return weighted_features

    def get_config(self):
        config = super(CardiacSpatialTemporalAttention, self).get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'reduction_ratio': self.reduction_ratio
        })
        return config
