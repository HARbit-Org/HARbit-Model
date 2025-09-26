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

def create_cnn_lstm_model(input_shape, num_classes, use_tcn=True):
    """
    CNN-only model for (timesteps, channels) e.g. (250, 3).
    - No RNNs -> avoids CudnnRNN.
    - Optionally adds TCN-style dilated residual blocks for long-range temporal context.
    """
    inp = layers.Input(shape=input_shape)  # (T, C)

    # Stem
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)     # T -> T/2
    x = layers.Dropout(0.1)(x)

    # Mid block
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)     # T -> T/4
    x = layers.Dropout(0.1)(x)

    # Wider features
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Optional TCN stack to capture long temporal dependencies (no RNN)
    if use_tcn:
        for d in [1, 2, 4, 8, 16]:
            x = _temporal_block(x, filters=256, k=5, d=d, dropout=0.1, causal=True)

    # Global aggregation (no sequence op)
    x = layers.GlobalAveragePooling1D()(x)

    # Classifier head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)

    logits = Dense(num_classes)(x)
    out = layers.Activation("softmax", dtype="float32")(logits)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=_cnn_cfg["learning_rate"]),
        loss=_cnn_cfg["loss"],
        metrics=_cnn_cfg["metrics"],
    )
    return model
