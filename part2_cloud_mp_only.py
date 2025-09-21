# part2_cloud_mp_only.py
import os, json, time, pathlib, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from pathlib import Path
MODELS_DIR = Path("cloud_optimized_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_RATIO = 0.1
ROOT = pathlib.Path(".").resolve()
(ROOT / "reports").mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def build_cnn(width_mult=1.0, dense_units=256):
    c1, c2, c3 = int(32 * width_mult), int(64 * width_mult), int(128 * width_mult)
    du = int(dense_units * width_mult)
    i = layers.Input((32, 32, 3))
    x = i
    for _ in range(2):
        x = layers.Conv2D(c1, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    for _ in range(2):
        x = layers.Conv2D(c2, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    for _ in range(2):
        x = layers.Conv2D(c3, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(du, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    o = layers.Dense(10)(x)  # logits
    return keras.Model(i, o)

def compile_model(m):
    m.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return m

def get_ds_fp16(bs=256):
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    xtr = (xtr.astype("float16") / np.float16(255.0))
    xte = (xte.astype("float16") / np.float16(255.0))
    n = int(len(xtr) * (1 - VAL_RATIO))
    x1, y1 = xtr[:n], ytr[:n]
    xval, yval = xtr[n:], ytr[n:]

    def make(x, y, train=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if train:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

    return make(x1, y1, True), make(xval, yval, False), make(xte, yte, False)

def main():
    mixed_precision.set_global_policy("mixed_float16")
    m = build_cnn()
    # cast logits back to fp32
    x = layers.Activation("linear", dtype="float32", name="logits_fp32")(m.outputs[0])
    m = keras.Model(m.inputs, x)
    compile_model(m)
    EP_MP = 15
    tr, val, te = get_ds_fp16()
    t0 = time.perf_counter()
    hist = m.fit(tr, validation_data=val, epochs=EP_MP, verbose=2)
    (Path("reports")/"history_mp.json").write_text(json.dumps(hist.history, indent=2))

    t1 = time.perf_counter()
    total = t1 - t0
    ev = m.evaluate(te, verbose=0, return_dict=True)
    out = {
        "mixed_precision": {
            "epochs": EP_MP,
            "total_time_s": round(total, 2),
            "time_per_epoch_s": round(total / EP_MP, 2),
            "test_loss": float(ev.get("loss", 0.0)),
            "test_accuracy": float(ev.get("accuracy", 0.0)),
        }
    }
    # 评估后保存（放在 evaluate() 之后更直观）
    m.save(MODELS_DIR / f"mp_cnn_ep{EP_MP}.keras")
    (ROOT / "reports" / "cloud_mp.json").write_text(json.dumps(out, indent=2))
    print(out)

if __name__ == "__main__":
    main()
