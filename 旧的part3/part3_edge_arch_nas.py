# part3_edge_arch_nas.py
import os, json, time, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------- Config ----------
SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
tf.keras.mixed_precision.set_global_policy("float32")

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)
EDGE_DIR = ROOT / "edge_optimized_models"; EDGE_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_PATH = "cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.keras"
BATCH = 32
EPOCHS_FULL = 12     # Depthwise-Student 完整训练轮数（老师建议 10–15）
EPOCHS_PROXY = 4     # NAS 快速评估轮数（3–5）
TEST_SAMPLES = 800
REP_SAMPLES = 400
NUM_THREADS = 1

def load_data():
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32") / 255.0
    xte = xte.astype("float32") / 255.0
    return (xtr, ytr.flatten()), (xte, yte.flatten())

def subset(arr, n): 
    idx = np.random.RandomState(SEED).choice(len(arr[0]), size=min(n, len(arr[0])), replace=False)
    return arr[0][idx], arr[1][idx]

def rep_dataset_fn(xtr):
    idx = np.random.RandomState(SEED).choice(len(xtr), size=REP_SAMPLES, replace=False)
    for i in idx:
        yield [xtr[i:i+1].astype("float32")]

def compile_model(m, lr=1e-3):
    m.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return m

def build_depthwise_student(width_mult=0.75, depth_mult=1.0, dense_units=192):
    # 结构： (DW3x3 + PW1x1)x2 -> MP -> (DW+PW)x2 -> MP -> (DW+PW)x2 -> GAP -> Dense
    c1, c2, c3 = [int(v * width_mult) for v in (32, 64, 128)]
    rep1 = max(1, int(2 * depth_mult))
    rep2 = max(1, int(2 * depth_mult))
    rep3 = max(1, int(2 * depth_mult))
    du = int(dense_units * width_mult)
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

def try_transfer_low_layers(dst, src):
    # 简单的“按名字匹配、形状一致就拷贝”
    src_w = {w.name: w for w in src.weights}
    assigned = 0
    for w in dst.weights:
        if w.name in src_w and w.shape == src_w[w.name].shape:
            w.assign(src_w[w.name])
            assigned += 1
    return assigned

def keras_to_tflite(model, quant="fp32", rep_data=None, int8_io=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: rep_data
        if int8_io:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    return converter.convert()

def write_tflite(buf: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)
    return path.stat().st_size / 1e6

def eval_tflite(path: Path, x, y, num_threads=NUM_THREADS, warmup=10, runs=200):
    itp = tf.lite.Interpreter(model_path=str(path), num_threads=num_threads)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]; out = itp.get_output_details()[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
    for _ in range(warmup):
        itp.set_tensor(inp["index"], dummy); itp.invoke()
    n = min(runs, len(x))
    start = time.perf_counter(); ok = 0
    for i in range(n):
        xi = x[i:i+1].astype(inp["dtype"])
        itp.set_tensor(inp["index"], xi); itp.invoke()
        pred = itp.get_tensor(out["index"]).argmax(-1)[0]
        ok += int(pred == y[i])
    dt = time.perf_counter() - start
    return {"latency_ms_per_sample": (dt/n)*1000, "accuracy": ok/n, "samples": n}

def evaluate_keras(m, x, y):
    out = m.evaluate(x, y, batch_size=BATCH, verbose=0, return_dict=True)
    acc = float(next((out[k] for k in out if "acc" in k), list(out.values())[-1]))
    return {"loss": float(out.get("loss", list(out.values())[0])), "accuracy": acc}

def train_with_earlystop(m, xtr, ytr, xval, yval, epochs):
    compile_model(m, lr=1e-3)
    cbs = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5)]
    hist = m.fit(xtr, ytr, validation_data=(xval, yval), batch_size=BATCH, epochs=epochs, verbose=2, callbacks=cbs)
    return hist.history

def main():
    (xtr, ytr), (xte, yte) = load_data()
    # 划验证集与评测集
    n = int(0.9*len(xtr))
    x_train, y_train = xtr[:n], ytr[:n]
    x_val, y_val = xtr[n:], ytr[n:]
    x_eval, y_eval = subset((xte, yte), TEST_SAMPLES)

    # 加载 student，用作迁移的来源
    student = keras.models.load_model(STUDENT_PATH)

    # === 1) Depthwise-Student 完整训练 ===
    dw = build_depthwise_student(width_mult=0.75, depth_mult=1.0, dense_units=192)
    assigned = try_transfer_low_layers(dw, student)  # 能迁就迁（形状相同）
    hist = train_with_earlystop(dw, x_train, y_train, x_val, y_val, epochs=EPOCHS_FULL)
    keras_eval = evaluate_keras(dw, x_eval, y_eval)
    # 保存 Keras
    dw_path = EDGE_DIR / f"depthwise_student_w0_75_d1_0.keras"
    dw.save(dw_path)

    # TFLite（FP32/FP16/INT8）
    rep = rep_dataset_fn(x_train)
    packs = {}
    for q in ["fp32", "fp16", "int8"]:
        buf = keras_to_tflite(dw, quant="dynamic" if q=="fp32" else q, rep_data=rep if q=="int8" else None)
        p = EDGE_DIR / f"depthwise_student_w0_75_d1_0_{q}.tflite"
        size = write_tflite(buf, p)
        e = eval_tflite(p, x_eval, y_eval, NUM_THREADS)
        packs[q] = {"size_mb": size, **e}

    results = {
        "depthwise_student": {
            "transferred_weights": int(assigned),
            "epochs": EPOCHS_FULL,
            "keras_eval": keras_eval,
            "tflite": packs
        },
        "nas": []
    }

    # === 2) 简易 NAS（小网格 + 短训） ===
    width_space = [0.5, 0.75, 1.0]
    depth_space = [1.0, 1.5]
    for w in width_space:
        for d in depth_space:
            m = build_depthwise_student(width_mult=w, depth_mult=d, dense_units=192)
            try_transfer_low_layers(m, student)
            h = train_with_earlystop(m, x_train, y_train, x_val, y_val, epochs=EPOCHS_PROXY)
            ke = evaluate_keras(m, x_eval, y_eval)
            # 用 FP16 作为边缘常用折中；INT8 也可以，但 PTQ 标定较慢
            buf = keras_to_tflite(m, quant="fp16")
            path = EDGE_DIR / f"nas_w{str(w).replace('.','_')}_d{str(d).replace('.','_')}_fp16.tflite"
            size = write_tflite(buf, path)
            te = eval_tflite(path, x_eval, y_eval, NUM_THREADS)
            results["nas"].append({
                "width_mult": w, "depth_mult": d,
                "epochs_proxy": EPOCHS_PROXY,
                "keras_accuracy": ke["accuracy"],
                "tflite_fp16_size_mb": size,
                "tflite_fp16_latency_ms": te["latency_ms_per_sample"],
                "tflite_fp16_accuracy": te["accuracy"]
            })

    (REPORTS / "edge_arch_nas.json").write_text(json.dumps(results, indent=2))
    print("[done] wrote", REPORTS / "edge_arch_nas.json")

if __name__ == "__main__":
    main()
