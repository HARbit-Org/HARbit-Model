from io import StringIO
from datetime import datetime
from pathlib import Path
import os, sys, gc, time
import yaml
import joblib
import numpy as np
import pandas as pd

# --- Paths / imports ---------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR.parent))  # if you need project root on PYTHONPATH

# Project utils
from utils import *                          # load_sensors_separately, etc.
# Use your existing model builder:
from models.main import create_cnn_lstm_model  # or swap for create_cnn_model

# --- TensorFlow setup ---------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import mixed_precision

gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except:
        pass

# Mixed precision: leave enabled if it worked well for you
mixed_precision.set_global_policy("mixed_float16")
tf.config.optimizer.set_jit(False)
print("GPUs:", gpus)

# --- Config / constants -------------------------------------------------------
ACT = SRC_DIR / "config" / "activities.yaml"
with open(ACT, "r") as f:
    _cfg = yaml.safe_load(f)["config"]
activities_ = _cfg["labels"]
cluster_    = _cfg["clusters"]

# Data & training params
DATA_DIR          = SRC_DIR / "data" / "wisdm-dataset" / "raw" / "watch"
WINDOW_SECONDS    = 5
OVERLAP_PERCENT   = 50
SAMPLING_RATE     = 20
TARGET_TIMESTEPS  = 100          # 5s * 20Hz -> 100 (you can set 250 if desired)
MIN_DATA_THRESH   = 0.8
MAX_GAP_SECONDS   = 1.0

EPOCHS            = 20
BATCH_SIZE        = 256
SHUFFLE_BUF       = 10000

# Output
OUT_ROOT = SRC_DIR / "artifacts"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR  = OUT_ROOT / f"fulltrain_{stamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Your windowing helpers (imported from your existing script) -------------
# If these are already in your codebase, import them instead of redefining.

def validate_window_data(window_data_df, window_seconds, sampling_rate, 
                         min_data_threshold, max_gap_seconds):
    if len(window_data_df) == 0:
        return False, {'reason': 'empty', 'data_coverage': 0, 'max_gap': float('inf'), 'actual_rate': 0}
    expected = window_seconds * sampling_rate
    actual   = len(window_data_df)
    cov      = actual / expected

    if cov < min_data_threshold:
        return False, {'reason':'insufficient_data','data_coverage':cov,'max_gap':float('inf'),'actual_rate':0}

    if len(window_data_df) > 1:
        timestamps  = pd.to_datetime(window_data_df['Timestamp'])
        time_diffs  = timestamps.diff().dt.total_seconds().fillna(0)
        max_gap     = float(time_diffs.max())
        actual_rate = len(window_data_df) / max(1e-9, (timestamps.max() - timestamps.min()).total_seconds())
    else:
        max_gap, actual_rate = 0.0, sampling_rate

    if max_gap > max_gap_seconds:
        return False, {'reason':'large_gap','data_coverage':cov,'max_gap':max_gap,'actual_rate':actual_rate}

    sensor = window_data_df[['X','Y','Z']].values
    if np.any(np.isnan(sensor)) or np.any(np.isinf(sensor)):
        return False, {'reason':'invalid_values','data_coverage':cov,'max_gap':max_gap,'actual_rate':actual_rate}

    return True, {'reason':'valid','data_coverage':cov,'max_gap':max_gap,'actual_rate':actual_rate}

def resample_window_robust(sensor_data, timestamps, target_timesteps, window_seconds):
    from scipy.interpolate import interp1d
    from scipy import signal
    if len(sensor_data) == 0:
        return np.zeros((target_timesteps, 3))
    n = len(sensor_data)
    if n == target_timesteps: return sensor_data.copy()
    if n == 1: return np.tile(sensor_data[0], (target_timesteps, 1))

    try:
        if hasattr(timestamps[0], 'timestamp'):
            t_sec = np.array([t.timestamp() for t in timestamps])
        elif isinstance(timestamps[0], pd.Timestamp):
            t_sec = np.array([t.timestamp() for t in timestamps])
        else:
            t_sec = timestamps.astype('int64') / 1e9
        t_min, t_max = t_sec.min(), t_sec.max()
        rel = (t_sec - t_min) / (t_max - t_min) if t_max > t_min else np.linspace(0,1,len(t_sec))
        tgt = np.linspace(0,1,target_timesteps)

        out = np.zeros((target_timesteps,3))
        for ax in range(3):
            try:
                if n >= target_timesteps:
                    out[:,ax] = signal.resample(sensor_data[:,ax], target_timesteps)
                else:
                    if len(np.unique(rel)) > 1:
                        itp = interp1d(rel, sensor_data[:,ax],
                                       kind='cubic' if n>=4 else 'linear',
                                       bounds_error=False, fill_value='extrapolate')
                        out[:,ax] = itp(tgt)
                    else:
                        out[:,ax] = np.full(target_timesteps, sensor_data[0,ax])
            except:
                out[:,ax] = np.interp(tgt, rel, sensor_data[:,ax])
        return out
    except Exception as e:
        print(f"Resample error: {e}")
        return np.tile(sensor_data[0], (target_timesteps, 1))

def is_window_quality_good(resampled_window, max_std_threshold=50.0):
    if np.any(np.isnan(resampled_window)) or np.any(np.isinf(resampled_window)):
        return False
    if np.any(np.abs(resampled_window) > 1000):
        return False
    for ax in range(resampled_window.shape[1]):
        s = float(np.std(resampled_window[:,ax]))
        if s > max_std_threshold or s < 1e-3:
            return False
    return True

def create_raw_windows_250_timesteps_robust(
    df, window_seconds=5, overlap_percent=50, sampling_rate=20,
    target_timesteps=250, min_data_threshold=0.5, max_gap_seconds=1.0
):
    print("🔧 Configuración de ventanas RAW ROBUSTA")
    if hasattr(df, 'to_pandas'):
        df_pd = df.to_pandas()
    else:
        df_pd = df.copy()

    if df_pd['Timestamp'].dtype == 'object' or str(df_pd['Timestamp'].dtype).startswith('datetime64') is False:
        df_pd['Timestamp'] = pd.to_datetime(df_pd['Timestamp'])

    window_ns = int(window_seconds * 1e9)
    step_ns   = int(window_ns * (100 - overlap_percent) / 100)

    X_windows, y_labels, subjects_list, metadata_list = [], [], [], []
    total_attempted = total_created = 0

    for (user_id, activity), group in df_pd.groupby(['Subject-id','Activity Label']):
        group = group.sort_values('Timestamp').dropna(subset=['X','Y','Z','Timestamp']).reset_index(drop=True)
        if len(group) < window_seconds * sampling_rate:
            continue

        t_ns = group['Timestamp'].astype('int64').values
        start_ns, end_ns = int(t_ns.min()), int(t_ns.max())
        current = start_ns
        window_idx = 0

        while current + window_ns <= end_ns:
            total_attempted += 1
            w_end = current + window_ns
            mask = (t_ns >= current) & (t_ns < w_end)
            wdf  = group.loc[mask]

            ok, info = validate_window_data(wdf, window_seconds, sampling_rate, min_data_threshold, max_gap_seconds)
            if ok:
                data = wdf[['X','Y','Z']].values
                ts   = wdf['Timestamp'].values
                try:
                    res = resample_window_robust(data, ts, target_timesteps, window_seconds)
                    if is_window_quality_good(res):
                        X_windows.append(res)
                        y_labels.append(activity)
                        subjects_list.append(user_id)
                        metadata_list.append({
                            'Subject-id': user_id,
                            'Activity Label': activity,
                            'window_start': pd.to_datetime(current),
                            'window_end':   pd.to_datetime(w_end),
                            'original_samples': len(wdf),
                            'resampled_timesteps': target_timesteps,
                            'window_idx': window_idx,
                            'actual_duration_s': window_seconds,
                            'data_coverage': info['data_coverage'],
                            'max_gap_s': info['max_gap'],
                            'sampling_rate_actual': info['actual_rate']
                        })
                        total_created += 1
                        window_idx += 1
                except Exception as e:
                    print(f"Ventana error (user {user_id}): {e}")

            current += step_ns

    if len(X_windows) == 0:
        print("❌ No se crearon ventanas válidas")
        return None, None, None, None

    X = np.array(X_windows)
    y = np.array(y_labels)
    subjects = np.array(subjects_list)
    meta = pd.DataFrame(metadata_list)

    print(f"✅ Ventanas: {len(X)}  | X shape: {X.shape}")
    return X, y, subjects, meta

# --- Load data ---------------------------------------------------------------
print("📥 Cargando datos...")
sensor_data = load_sensors_separately(DATA_DIR)
df_accel = sensor_data["accel"].to_pandas()

print(f"Acelerómetro: {len(df_accel)} muestras")

# --- Build windows -----------------------------------------------------------
X_all, y_all_raw, subjects_all, metadata_all = create_raw_windows_250_timesteps_robust(
    df=df_accel,
    window_seconds=WINDOW_SECONDS,
    overlap_percent=OVERLAP_PERCENT,
    sampling_rate=SAMPLING_RATE,
    target_timesteps=TARGET_TIMESTEPS,
    min_data_threshold=MIN_DATA_THRESH,
    max_gap_seconds=MAX_GAP_SECONDS
)
if X_all is None:
    raise SystemExit("No windows created. Aborting.")

# --- Map labels -> activities -> clusters -----------------------------------
print("🔤 Mapeando etiquetas a clusters...")
y_all_mapped = []
for lbl in y_all_raw:
    act_name = activities_[lbl]                  # map id/code -> activity string
    cluster_name = act_name
    for c_name, acts in cluster_.items():
        if act_name in acts:
            cluster_name = c_name
            break
    y_all_mapped.append(cluster_name)
y_all = np.array(y_all_mapped)

# --- Encode labels -----------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y_all)
class_names = le.classes_.tolist()
num_classes = len(class_names)
print(f"Clases ({num_classes}): {class_names}")

# --- TF dataset (train on ALL data) -----------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
X_all = X_all.astype("float32")
y_enc = y_enc.astype("int32")

ds = (tf.data.Dataset.from_tensor_slices((X_all, y_enc))
      .shuffle(SHUFFLE_BUF)
      .batch(BATCH_SIZE)
      .prefetch(AUTOTUNE))

# --- Build & train model -----------------------------------------------------
input_shape = (TARGET_TIMESTEPS, X_all.shape[2])  # (T, channels), channels=3 for accel
model = create_cnn_lstm_model(input_shape=input_shape, num_classes=num_classes)
model.summary()

print("🏋️ Entrenando con TODO el dataset de ventanas...")
history = model.fit(ds, epochs=EPOCHS, verbose=1)

# --- Save artifacts ----------------------------------------------------------
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(OUT_DIR / "history.csv", index=False)

# Save label encoder & class names
joblib.dump(le, OUT_DIR / "label_encoder.joblib")
with open(OUT_DIR / "classes.txt", "w") as f:
    for c in class_names:
        f.write(f"{c}\n")

# Save model as .h5 (requires h5py installed)
model_h5 = OUT_DIR / "model.h5"
tf.keras.models.save_model(model, model_h5)  # .h5 extension triggers HDF5 saving

# Also save a native Keras format (optional, nice to have)
model_keras = OUT_DIR / "model.keras"
model.save(model_keras)

print("\n✅ Entrenamiento completo.")
print(f"📦 Artefactos guardados en: {OUT_DIR}")
print(f"   - {model_h5.name}")
print(f"   - {model_keras.name}")
print(f"   - history.csv")
print(f"   - label_encoder.joblib")
print(f"   - classes.txt")
