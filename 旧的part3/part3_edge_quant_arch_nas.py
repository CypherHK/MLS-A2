# part3_edge_quant_arch_nas.py
# 环境: TF 2.16.x + Keras 3.x（CPU/Metal；禁混合精度）
import argparse, json, time, random
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras import layers

# --------- Defaults (可被 CLI 覆盖) ----------
SEED = 42
DEFAULT_BATCH = 32
DEFAULT_EPOCHS_FULL = 12
DEFAULT_EPOCHS_PROXY = 4
DEFAULT_TEST_SAMPLES = 1000
DEFAULT_REP_SAMPLES = 400
DEFAULT_NUM_THREADS = 1
DEFAULT_WARMUP = 10
DEFAULT_RUNS_PER_MODEL = 200
DEFAULT_WIDTH_SPACE = [0.5, 0.75, 1.0]
DEFAULT_DEPTH_SPACE = [1.0, 1.5]

# --------- Setup ----------
np.random.seed(SEED); random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
keras.mixed_precision.set_global_policy("float32")  # 禁混合精度

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)
EDGE_DIR = ROOT / "edge_optimized_models"; EDGE_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_PATH = "cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.keras"

# --------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Edge PTQ / Depthwise / NAS (with smoke mode)")
    p.add_argument("--smoke", action="store_true", help="快速冒烟：极少 epoch、缩小样本、缩小 NAS 搜索空间，并默认跳过 INT8")
    p.add_argument("--no-int8", action="store_true", help="禁用 INT8 量化（无代表集或想加速时很有用）")
    p.add_argument("--epochs-full", type=int, default=DEFAULT_EPOCHS_FULL)
    p.add_argument("--epochs-proxy", type=int, default=DEFAULT_EPOCHS_PROXY)
    p.add_argument("--test-samples", type=int, default=DEFAULT_TEST_SAMPLES)
    p.add_argument("--rep-samples", type=int, default=DEFAULT_REP_SAMPLES)
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS_PER_MODEL, help="TFLite评测总推理数（<= test-samples）")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--threads", type=int, default=DEFAULT_NUM_THREADS)
    p.add_argument("--width-space", type=str, default=",".join(map(str, DEFAULT_WIDTH_SPACE)),
                   help="NAS 宽度倍率，逗号分隔，如: 0.5,0.75,1.0")
    p.add_argument("--depth-space", type=str, default=",".join(map(str, DEFAULT_DEPTH_SPACE)),
                   help="NAS 深度倍率，逗号分隔，如: 1.0,1.5")
    return p.parse_args()

# --------- Data ----------
def load_data():
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32") / 255.0
    xte = xte.astype("float32") / 255.0
    return (xtr, ytr.flatten()), (xte, yte.flatten())

def subset(arr, n):
    x, y = arr
    n = min(n, len(x))
    return x[:n], y[:n]

def rep_dataset_fn(xtr, rep_samples, seed=SEED):
    idx = np.random.RandomState(seed).choice(len(xtr), size=min(rep_samples, len(xtr)), replace=False)
    for i in idx:
        yield [xtr[i:i+1].astype("float32")]

# --------- Models ----------
def compile_model(m, lr=1e-3, batch=DEFAULT_BATCH):
    m.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return m

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

def try_transfer_low_layers(dst, src):
    name_to_w = {w.name: w for w in src.weights}
    assigned = 0
    for w in dst.weights:
        if w.name in name_to_w and tuple(w.shape) == tuple(name_to_w[w.name].shape):
            w.assign(name_to_w[w.name]); assigned += 1
    return assigned

# --------- TFLite ----------
def keras_to_tflite(model, quant="fp32", rep_data=None, int8_io=False, force_micro_int8=False):
    """
    quant: "fp32" (dynamic), "fp16", "int8"
    int8_io: True -> I/O 也是 int8（TFLM 常用）
    force_micro_int8: True -> 强制使用 TFLM 支持的全 INT8 内核集
    """
    def _apply_quant(converter):
        if quant == "fp32":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quant == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quant == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if rep_data is None:
                raise ValueError("INT8 量化需要 representative dataset")
            converter.representative_dataset = lambda: rep_data
            if force_micro_int8:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            if int8_io or force_micro_int8:
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        return converter.convert()

    # A) 直接从 Keras 模型（最稳）
    try:
        return _apply_quant(tf.lite.TFLiteConverter.from_keras_model(model))
    except Exception as e1:
        # B) 从 ConcreteFunction 回退
        try:
            @tf.function(jit_compile=False)
            def f(x):
                return model(x, training=False)
            concrete = f.get_concrete_function(tf.TensorSpec([1,32,32,3], tf.float32))
            return _apply_quant(tf.lite.TFLiteConverter.from_concrete_functions([concrete], model))
        except Exception as e2:
            # C) 再回退到 SavedModel（最后一招）
            import shutil
            tmp = EDGE_DIR / "_tmp_savedmodel_for_tflite"
            if tmp.exists(): shutil.rmtree(tmp)
            model.save(tmp, save_format="tf")
            try:
                return _apply_quant(tf.lite.TFLiteConverter.from_saved_model(str(tmp)))
            finally:
                try: shutil.rmtree(tmp)
                except Exception: pass


def write_tflite(buf: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)
    return path.stat().st_size / 1e6

def _quantize_to_input_dtype(x1, inp_detail):
    dtype = inp_detail["dtype"]
    if dtype == np.float32:
        return x1.astype(np.float32)
    scale, zp = inp_detail.get("quantization", (0.0, 0))
    if not scale:
        return x1.astype(dtype)
    q = np.round(x1 / scale + zp)
    q = np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return q

def eval_tflite(path: Path, x, y, num_threads=DEFAULT_NUM_THREADS, warmup=DEFAULT_WARMUP, runs=DEFAULT_RUNS_PER_MODEL):
    itp = tf.lite.Interpreter(model_path=str(path), num_threads=num_threads)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]; out = itp.get_output_details()[0]
    dummy = np.zeros(inp["shape"], dtype=inp["dtype"])
    for _ in range(warmup):
        itp.set_tensor(inp["index"], dummy); itp.invoke()
    n = min(runs, len(x))
    start = time.perf_counter(); ok = 0
    for i in range(n):
        xi = x[i:i+1].astype("float32")
        xi = _quantize_to_input_dtype(xi, inp)
        itp.set_tensor(inp["index"], xi); itp.invoke()
        pred = int(itp.get_tensor(out["index"]).argmax(-1))
        ok += int(pred == int(y[i]))
    dt = time.perf_counter() - start
    return {"latency_ms_per_sample": (dt/n)*1000, "accuracy": ok/n, "samples": n}

def evaluate_keras(m, x, y, batch=DEFAULT_BATCH):
    out = m.evaluate(x, y, batch_size=batch, verbose=0, return_dict=True)
    acc = float(next((out[k] for k in out if "acc" in k), list(out.values())[-1]))
    return {"loss": float(out.get("loss", list(out.values())[0])), "accuracy": acc}

def train_with_earlystop(m, xtr, ytr, xval, yval, epochs, batch=DEFAULT_BATCH):
    compile_model(m, lr=1e-3, batch=batch)
    cbs = [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5)]
    hist = m.fit(xtr, ytr, validation_data=(xval, yval),
                 batch_size=batch, epochs=epochs, verbose=2, callbacks=cbs)
    return hist.history

# --------- Main ----------
def main():
    args = parse_args()

    # 解析搜索空间
    width_space = [float(s) for s in args.width_space.split(",") if s]
    depth_space = [float(s) for s in args.depth_space.split(",") if s]

    # 冒烟模式：极简配置加速验证流程
    if args.smoke:
        epochs_full = 2
        epochs_proxy = 1
        test_samples = min(200, args.test_samples)
        rep_samples = min(64, args.rep_samples)
        runs = min(50, args.runs)
        warmup = min(2, args.warmup)
        threads = args.threads
        width_space = [0.5]  # 缩小 NAS
        depth_space = [1.0]
        quant_list = ["fp32", "fp16"]  # 默认跳过 INT8
    else:
        epochs_full = args.epochs_full
        epochs_proxy = args.epochs_proxy
        test_samples = args.test_samples
        rep_samples = args.rep_samples
        runs = args.runs
        warmup = args.warmup
        threads = args.threads
        quant_list = ["fp32", "fp16", "int8"]

    if args.no_int8 and "int8" in quant_list:
        quant_list.remove("int8")

    (xtr, ytr), (xte, yte) = load_data()
    # 划验证/评测子集
    n = int(0.9 * len(xtr))
    x_train, y_train = xtr[:n], ytr[:n]
    x_val, y_val = xtr[n:], ytr[n:]
    x_eval, y_eval = subset((xte, yte), test_samples)

    # 加载 Student (Keras3) 作为迁移来源
    student = keras.models.load_model(STUDENT_PATH)

    # === 1) Depthwise-Student 训练 & PTQ ===
    dw = build_depthwise_student(width_mult=0.75, depth_mult=1.0, dense_units=192)
    assigned = try_transfer_low_layers(dw, student)
    _ = train_with_earlystop(dw, x_train, y_train, x_val, y_val, epochs=epochs_full)
    keras_eval = evaluate_keras(dw, x_eval, y_eval)

    dw_path = EDGE_DIR / f"depthwise_student_w0_75_d1_0.keras"
    dw.save(dw_path)
     
    packs = {}
    for q in quant_list:
        try:
            rep = rep_dataset_fn(x_train, rep_samples) if q == "int8" else None
            buf = keras_to_tflite(dw, quant=q, rep_data=rep, int8_io=False)
            p = EDGE_DIR / f"depthwise_student_w0_75_d1_0_{q}.tflite"
            size = write_tflite(buf, p)
            e = eval_tflite(p, x_eval, y_eval, num_threads=threads, warmup=warmup, runs=runs)
            packs[q] = {"size_mb": size, **e, "tflite_path": str(p)}
        except Exception as ex:
            packs[q] = {"error": str(ex)}

    results = {
        "config": {
            "batch": DEFAULT_BATCH,
            "epochs_full": epochs_full,
            "epochs_proxy": epochs_proxy,
            "rep_samples": rep_samples,
            "num_threads": threads,
            "warmup": warmup,
            "runs_per_model": runs,
            "test_samples": test_samples,
            "quant_list": quant_list
        },
        "depthwise_student": {
            "transferred_weights": int(assigned),
            "keras_eval": keras_eval,
            "keras_path": str(dw_path),
            "tflite": packs
        },
        "nas": []
    }

    # === 2) 简易 NAS（短训 + FP16 评测） ===
    for w in width_space:
        for d in depth_space:
            m = build_depthwise_student(width_mult=w, depth_mult=d, dense_units=192)
            try_transfer_low_layers(m, student)
            _ = train_with_earlystop(m, x_train, y_train, x_val, y_val, epochs=epochs_proxy)
            ke = evaluate_keras(m, x_eval, y_eval)
            try:
                buf = keras_to_tflite(m, quant="fp16")
                path = EDGE_DIR / f"nas_w{str(w).replace('.','_')}_d{str(d).replace('.','_')}_fp16.tflite"
                size = write_tflite(buf, path)
                te = eval_tflite(path, x_eval, y_eval, num_threads=threads, warmup=warmup, runs=runs)
                results["nas"].append({
                    "width_mult": float(w), "depth_mult": float(d),
                    "epochs_proxy": epochs_proxy,
                    "keras_accuracy": ke["accuracy"],
                    "tflite_fp16_path": str(path),
                    "tflite_fp16_size_mb": size,
                    "tflite_fp16_latency_ms": te["latency_ms_per_sample"],
                    "tflite_fp16_accuracy": te["accuracy"]
                })
            except Exception as ex:
                results["nas"].append({
                    "width_mult": float(w), "depth_mult": float(d),
                    "epochs_proxy": epochs_proxy,
                    "keras_accuracy": ke["accuracy"],
                    "tflite_fp16_error": str(ex)
                })

    (REPORTS / "edge_quant_arch_nas.json").write_text(json.dumps(results, indent=2))
    print("[done] wrote", REPORTS / "edge_quant_arch_nas.json")

if __name__ == "__main__":
    main()
