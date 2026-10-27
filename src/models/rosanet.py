"""
RosaNet Feature Extractor and Recording-Level Classifier for Heart Sound Auscultation.

Architecture details:
1. Input: Log-Mel Spectrogram (64 Mel frequency bands x Time steps x 1 channel)
2. Multi-Path Receptive Field Convolutional Block:
   - Branch A: 3x3 Conv (fine temporal/spectral detail, e.g. sharp valve clicks / S1/S2 splits)
   - Branch B: 5x5 Conv (intermediate murmur patterns)
   - Branch C: 7x7 Conv (broad resonance / low-frequency rumbling murmurs)
3. Convolutional Feature Extraction Blocks with Batch Normalization, ReLU, Max Pooling, and Dropout.
4. Split-Self-Attention integrated at deep convolutional stage.
5. Global Feature Pooling (GAP + GMP) -> 128-D Feature Embedding.
6. Temporal/Channel Feature Attention -> 128-D Weighted Feature Vector.
7. Recording-level Disease Prediction Head: Dense(5, activation='sigmoid') for [N, AS, AR, MR, MS].
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from src.models.attention import SplitSelfAttention, CardiacSpatialTemporalAttention


def build_multipath_block(x: tf.Tensor, filters_per_branch: int = 16, name_prefix: str = "mp_block") -> tf.Tensor:
    """
    Multi-path convolutional block capturing diverse temporal and spectral receptive fields.
    """
    # Path 1: 3x3 Conv
    p1 = layers.Conv2D(filters_per_branch, (3, 3), padding='same', name=f"{name_prefix}_p1_conv")(x)
    p1 = layers.BatchNormalization(name=f"{name_prefix}_p1_bn")(p1)
    p1 = layers.ReLU(name=f"{name_prefix}_p1_relu")(p1)

    # Path 2: 5x5 Conv
    p2 = layers.Conv2D(filters_per_branch, (5, 5), padding='same', name=f"{name_prefix}_p2_conv")(x)
    p2 = layers.BatchNormalization(name=f"{name_prefix}_p2_bn")(p2)
    p2 = layers.ReLU(name=f"{name_prefix}_p2_relu")(p2)

    # Path 3: 7x7 Conv
    p3 = layers.Conv2D(filters_per_branch, (7, 7), padding='same', name=f"{name_prefix}_p3_conv")(x)
    p3 = layers.BatchNormalization(name=f"{name_prefix}_p3_bn")(p3)
    p3 = layers.ReLU(name=f"{name_prefix}_p3_relu")(p3)

    # Path 4: 1x1 Conv (pointwise baseline)
    p4 = layers.Conv2D(filters_per_branch, (1, 1), padding='same', name=f"{name_prefix}_p4_conv")(x)
    p4 = layers.BatchNormalization(name=f"{name_prefix}_p4_bn")(p4)
    p4 = layers.ReLU(name=f"{name_prefix}_p4_relu")(p4)

    # Concatenate all parallel paths
    concat = layers.concatenate([p1, p2, p3, p4], axis=-1, name=f"{name_prefix}_concat")
    return concat


def build_rosanet_backbone(
    input_shape: tuple = (64, 626, 1),
    embedding_dim: int = 128,
    dropout_rate: float = 0.3,
    model_name: str = "rosanet_backbone"
) -> tf.keras.Model:
    """
    Builds the standalone RosaNet feature extractor backbone.
    
    Input: (B, 64, time_steps, 1)
    Output: (B, embedding_dim) weighted feature representation.
    """
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram_input")
    
    # Block 1: Multipath Convolutional Stage (Total filters = 4 * 16 = 64)
    x = build_multipath_block(inputs, filters_per_branch=16, name_prefix="block1_mp")
    x = layers.MaxPooling2D(pool_size=(2, 2), name="block1_pool")(x)
    x = layers.Dropout(dropout_rate * 0.5, name="block1_dropout")(x)

    # Block 2: Intermediate Convolutional Stage (64 filters)
    x = layers.Conv2D(64, (3, 3), padding='same', name="block2_conv1")(x)
    x = layers.BatchNormalization(name="block2_bn1")(x)
    x = layers.ReLU(name="block2_relu1")(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name="block2_conv2")(x)
    x = layers.BatchNormalization(name="block2_bn2")(x)
    x = layers.ReLU(name="block2_relu2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="block2_pool")(x)
    x = layers.Dropout(dropout_rate * 0.7, name="block2_dropout")(x)

    # Block 3: Deep Feature Stage with Split-Self-Attention (128 filters)
    x = layers.Conv2D(128, (3, 3), padding='same', name="block3_conv1")(x)
    x = layers.BatchNormalization(name="block3_bn1")(x)
    x = layers.ReLU(name="block3_relu1")(x)
    
    # Split-Self-Attention mechanism at deeper layer
    x = SplitSelfAttention(channels=128, name="block3_split_self_attention")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="block3_pool")(x)
    x = layers.Dropout(dropout_rate, name="block3_dropout")(x)

    # Block 4: High-Level Feature Stage (128 filters)
    x = layers.Conv2D(128, (3, 3), padding='same', name="block4_conv1")(x)
    x = layers.BatchNormalization(name="block4_bn1")(x)
    x = layers.ReLU(name="block4_relu1")(x)

    # Global Hybrid Feature Pooling (GAP + GMP)
    gap = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    gmp = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    pooled = layers.concatenate([gap, gmp], name="hybrid_global_pool")
    
    # Dense Projection to embedding dimension
    embeddings = layers.Dense(embedding_dim, name="feature_embedding_dense")(pooled)
    embeddings = layers.BatchNormalization(name="feature_embedding_bn")(embeddings)
    embeddings = layers.ReLU(name="feature_embedding_relu")(embeddings)
    
    # Attention mechanism on learned feature vector
    weighted_features = CardiacSpatialTemporalAttention(
        feature_dim=embedding_dim,
        reduction_ratio=4,
        name="feature_importance_attention"
    )(embeddings)

    return models.Model(inputs=inputs, outputs=weighted_features, name=model_name)


def build_rosanet_recording_classifier(
    input_shape: tuple = (64, 626, 1),
    num_classes: int = 5,
    embedding_dim: int = 128,
    dropout_rate: float = 0.3,
    model_name: str = "rosanet_recording_classifier"
) -> tf.keras.Model:
    """
    Builds a single-recording RosaNet classifier.
    
    Input: (B, 64, time_steps, 1)
    Outputs:
        - predictions: (B, num_classes) in [0, 1] (multi-label sigmoid)
        - features: (B, embedding_dim) weighted feature vector
    """
    inputs = layers.Input(shape=input_shape, name="single_spectrogram_input")
    backbone = build_rosanet_backbone(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        model_name=f"{model_name}_backbone"
    )
    
    features = backbone(inputs)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(features)
    x = layers.Dense(64, activation='relu', name="classifier_dense")(x)
    predictions = layers.Dense(num_classes, activation='sigmoid', name="recording_prediction")(x)

    return models.Model(inputs=inputs, outputs=[predictions, features], name=model_name)
