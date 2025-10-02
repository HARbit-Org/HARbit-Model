import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, LayerNormalization, RNN, LSTMCell
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from pathlib import Path
from sklearn.preprocessing import (
    LabelEncoder, 
    StandardScaler,
    label_binarize
)

import yaml

MODELS_DIR = Path(__file__).resolve().parents[1]
CFG = MODELS_DIR / "config" / "hiperparameters.yaml"

with open(CFG, "r") as f:
    config = yaml.safe_load(f)["config"]

_cnn_lstm_config = config['cnn-lstm']

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from pathlib import Path
import yaml

# ---- config ----
MODELS_DIR = Path(__file__).resolve().parents[1]
CFG = MODELS_DIR / "config" / "hiperparameters.yaml"

with open(CFG, "r") as f:
    config = yaml.safe_load(f)["config"]

# Prefer a "cnn" section; fall back to your existing "cnn-lstm" if that's what you have
_cnn_cfg = config.get("cnn", config.get("cnn-lstm", {
    "learning_rate": 1e-3,
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
}))

# ---- optional: small residual temporal block (TCN-style) ----
def _temporal_block(x, filters, k=5, d=1, dropout=0.1, causal=True):
    pad = "causal" if causal else "same"
    y = layers.Conv1D(filters, k, padding=pad, dilation_rate=d, activation="relu")(x)
    y = layers.LayerNormalization()(y)
    y = layers.Conv1D(filters, k, padding=pad, dilation_rate=d, activation="relu")(y)
    y = layers.LayerNormalization()(y)
    y = layers.Dropout(dropout)(y)
    # residual to match channels
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding="same")(x)
    return layers.Add()([x, y])

def create_cnn_lstm_with_features(input_shape_raw, input_shape_feat, num_classes, use_tcn=True):
    """
    Modelo multimodal: 
    - input 1: secuencias crudas (timesteps, channels)
    - input 2: features de ingeniería (num_features)
    """
    # --- Branch A: señales crudas ---
    inp_raw = layers.Input(shape=input_shape_raw, name="raw_input")  # (T, C)

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp_raw)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Optional TCN
    if use_tcn:
        for d in [1, 2, 4, 8, 16]:
            x = _temporal_block(x, filters=256, k=5, d=d, dropout=0.1, causal=True)

    x_raw = layers.GlobalAveragePooling1D()(x)

    # --- Branch B: features ---
    inp_feat = layers.Input(shape=input_shape_feat, name="feature_input")  # (num_features,)
    y = layers.Dense(128, activation="relu")(inp_feat)
    y = layers.LayerNormalization()(y)
    y = layers.Dropout(0.3)(y)

    # --- Fusion ---
    z = layers.Concatenate()([x_raw, y])
    z = layers.Dense(256, activation="relu")(z)
    z = layers.LayerNormalization()(z)
    z = layers.Dropout(0.3)(z)

    z = layers.Dense(128, activation="relu")(z)
    z = layers.LayerNormalization()(z)
    z = layers.Dropout(0.3)(z)

    logits = layers.Dense(num_classes)(z)
    out = layers.Activation("softmax", dtype="float32")(logits)

    model = models.Model(inputs=[inp_raw, inp_feat], outputs=out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=_cnn_cfg["learning_rate"]),
        loss=_cnn_cfg["loss"],
        metrics=_cnn_cfg["metrics"],
    )
    return model