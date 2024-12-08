import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, MultiHeadAttention)
from tensorflow.keras.optimizers import Adam

def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    # Multi-head Self-Attention
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed-Forward
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(inputs.shape[-1], activation='linear')(ff)
    ff = Dropout(dropout)(ff)
    out = LayerNormalization(epsilon=1e-6)(ff + x)

    return out

def create_classification_model(input_shape):
    inputs = Input(shape=input_shape)

    # Transformer encoder block
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)  # Optional second encoder layer

    # Global average pooling
    x = tf.reduce_mean(x, axis=1)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
