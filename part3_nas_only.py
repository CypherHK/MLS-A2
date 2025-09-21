# nas_only.py  (run in: mls-a2 with TF>=2.16 + Keras 3)
import json, random
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras import layers

SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
keras.mixed_precision.set_global_policy("float32")

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)

EPOCHS_PROXY = 3
BATCH = 32
TEST_SAMPLES = 1000
WIDTH_SPACE = [0.5, 0.75, 1.0]
DEPTH_SPACE = [1.0, 1.5]

def build_depthwise_student(width_mult=0.75, depth_mult=1.0, dense_units=192):
    c1, c2, c3 = [max(4, int(v * width_mult)) for v in (32, 64, 128)]
    rep = lambda: max(1, int(2 * depth_mult))
    rep1 = rep(); rep2 = rep(); rep3 = rep()
    du = max(16, int(dense_units * width_mult))

    i = layers.Input((32,32,3))
    x = i
    for _ in range(rep1):
        x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(c1, 1, padding="same", use_bias=False)(x);      x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    for _ in range(rep2):
        x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(c2, 1, padding="same", use_bias=False)(x);      x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    for _ in range(rep3):
        x = layers.DepthwiseConv2D(3, padding="same", use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(c3, 1, padding="same", use_bias=False)(x);      x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(du, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    o = layers.Dense(10)(x)  # logits
    return keras.Model(i, o)

def compile_model(m, lr=1e-3):
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return m

def load_cifar10():
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32")/255.0; xte = xte.astype("float32")/255.0
    return (xtr, ytr.flatten()), (xte, yte.flatten())

def subset(x, y, n): n=min(n,len(x)); return x[:n], y[:n]

def main():
    (xtr, ytr), (xte, yte) = load_cifar10()
    n = int(0.9*len(xtr))
    x_train, y_train = xtr[:n], ytr[:n]
    x_val, y_val = xtr[n:], ytr[n:]
    x_eval, y_eval = subset(xte, yte, TEST_SAMPLES)

    results = {"config":{"epochs_proxy":EPOCHS_PROXY,"batch":BATCH,
                         "width_space":WIDTH_SPACE,"depth_space":DEPTH_SPACE},
               "candidates":[]}

    for w in WIDTH_SPACE:
        for d in DEPTH_SPACE:
            m = build_depthwise_student(w, d)
            compile_model(m)
            m.fit(x_train, y_train, validation_data=(x_val, y_val),
                  epochs=EPOCHS_PROXY, batch_size=BATCH, verbose=2)
            ev = m.evaluate(x_eval, y_eval, batch_size=BATCH, verbose=0, return_dict=True)
            acc = float(ev.get("accuracy", list(ev.values())[-1]))
            params = int(np.sum([np.prod(v.shape) for v in m.weights]))
            results["candidates"].append({"width_mult":float(w), "depth_mult":float(d),
                                          "epochs_proxy":EPOCHS_PROXY, "eval_acc":acc,
                                          "params":params})

    (REPORTS/"nas_search_results.json").write_text(json.dumps(results, indent=2))
    print("[done] wrote", REPORTS/"nas_search_results.json")

if __name__ == "__main__":
    main()
