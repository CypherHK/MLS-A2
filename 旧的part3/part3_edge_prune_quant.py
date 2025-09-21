# part3_edge_prune_quant.py
import os, json, time, random
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import layers

# ----------- Config -----------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
tf.keras.mixed_precision.set_global_policy("float32")  # 避免 M1 MPSGraph 的 dtype 坑

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)
EDGE_DIR = ROOT / "edge_optimized_models"; EDGE_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_PATH = "cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.keras"
BATCH = 32
FT_EPOCHS = 8        # 剪枝微调轮数（老师建议：5–10）
REP_SAMPLES = 400    # INT8 标定代表性样本数量（200–500 都行）
TEST_SAMPLES = 1000  # 评测抽样数量（可加大/减小）
NUM_THREADS = 1      # TFLite 推理线程
SPARSITIES = [0.50, 0.75, 0.90]

# ----------- Data -----------
def load_data():
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32") / 255.0
    xte = xte.astype("float32") / 255.0
    return (xtr, ytr.flatten()), (xte, yte.flatten())

def rep_dataset_fn(xtr):
    idx = np.random.RandomState(SEED).choice(len(xtr), size=REP_SAMPLES, replace=False)
    for i in idx:
        # 代表性数据必须是 float32，形状 [1,H,W,C]
        yield [xtr[i:i+1].astype("float32")]

def subset_test(xte, yte):
    idx = np.random.RandomState(SEED + 1).choice(len(xte), size=min(TEST_SAMPLES, len(xte)), replace=False)
    return xte[idx], yte[idx]

# ----------- Utils -----------
def compile_model(m, lr=1e-3):
    m.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return m

def evaluate_keras(m, x, y, batch=BATCH):
    out = m.evaluate(x, y, batch_size=batch, verbose=0, return_dict=True)
    return {"test_loss": float(out.get("loss", list(out.values())[0])),
            "test_accuracy": float(next((out[k] for k in out if "acc" in k), list(out.values())[-1]))}

def save_keras(m, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(path)

def count_params(m): return int(m.count_params())

def tflite_from_keras(model, quant="fp32", rep_data=None, int8_io=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quant == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if rep_data is None:
            raise ValueError("INT8 quantization requires representative dataset.")
        converter.representative_dataset = lambda: rep_data
        if int8_io:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    return tflite_model

def write_tflite(buf: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf)
    return path.stat().st_size / 1e6  # MB

def eval_tflite(tflite_path: Path, x, y, num_threads=NUM_THREADS, warmup=10, runs=200):
    # 仅评测延迟与精度（抽样数据）
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    # 预热
    dummy = np.zeros(input_details["shape"], dtype=input_details["dtype"])
    for _ in range(warmup):
        interpreter.set_tensor(input_details["index"], dummy)
        interpreter.invoke()
    # 延迟
    start = time.perf_counter()
    n = min(runs, len(x))
    correct = 0
    for i in range(n):
        xi = x[i:i+1]
        # 若模型是 int8 输入，TFLite 会根据 scale/zero_point 完成量化，不需要你手动缩放
        xi = xi.astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], xi)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details["index"])
        pred = logits.argmax(axis=-1)
        correct += int(pred[0] == y[i])
    total = time.perf_counter() - start
    return {
        "latency_ms_per_sample": (total / n) * 1000.0,
        "tflite_eval_samples": n,
        "tflite_accuracy": correct / n
    }

# ----------- Pruning -----------
def prune_and_finetune(base_model, sparsity, xtr, ytr, xval, yval):
    end_step = np.ceil(len(xtr) / BATCH).astype(np.int32) * FT_EPOCHS
    schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=float(sparsity), begin_step=0, end_step=end_step
    )
    prunable = tfmot.sparsity.keras.prune_low_magnitude(base_model, pruning_schedule=schedule)
    compile_model(prunable, lr=5e-4)
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5)
    ]
    prunable.fit(xtr, ytr, validation_data=(xval, yval), epochs=FT_EPOCHS, batch_size=BATCH, verbose=2, callbacks=callbacks)
    # strip pruning wrapper
    stripped = tfmot.sparsity.keras.strip_pruning(prunable)
    return stripped

# ----------- Pipeline -----------
def main():
    (xtr, ytr), (xte, yte) = load_data()
    # 划一小块验证集
    n = int(len(xtr) * 0.9)
    x_train, y_train = xtr[:n], ytr[:n]
    x_val, y_val = xtr[n:], ytr[n:]
    x_eval, y_eval = subset_test(xte, yte)

    # 加载 student 作为起点
    student = keras.models.load_model(STUDENT_PATH)
    compile_model(student)
    base_eval = evaluate_keras(student, x_eval, y_eval)
    base_params = count_params(student)
    save_keras(student, EDGE_DIR / "student_base_fp32.keras")

    results = {
        "baseline_student": {
            "params": base_params,
            **base_eval
        },
        "pruning": {},
        "quantization": {}
    }

    # —— 剪枝多点 —— #
    best_key, best_acc = None, -1.0
    for sp in SPARSITIES:
        pruned = prune_and_finetune(keras.models.clone_model(student), sp, x_train, y_train, x_val, y_val)
        compile_model(pruned)
        eval_pruned = evaluate_keras(pruned, x_eval, y_eval)
        params_pruned = count_params(pruned)
        tag = f"sparsity_{int(sp*100)}"
        save_keras(pruned, EDGE_DIR / f"pruned_{tag}.keras")
        # 转换为 FP32 TFLite（未量化，仅对比）
        tflite_buf = tflite_from_keras(pruned, quant="fp32")
        size_mb = write_tflite(tflite_buf, EDGE_DIR / f"pruned_{tag}_fp32.tflite")
        tflite_eval = eval_tflite(EDGE_DIR / f"pruned_{tag}_fp32.tflite", x_eval, y_eval, NUM_THREADS)

        results["pruning"][tag] = {
            "params": params_pruned,
            **eval_pruned,
            "tflite_fp32_size_mb": size_mb,
            "tflite_fp32_latency_ms": tflite_eval["latency_ms_per_sample"],
            "tflite_fp32_accuracy": tflite_eval["tflite_accuracy"],
        }
        if eval_pruned["test_accuracy"] > best_acc:
            best_acc, best_key = eval_pruned["test_accuracy"], tag

    # —— 量化：对 student_base + 最优剪枝 各做 Dynamic/FP16/INT8 —— #
    def quant_pack(model, name_prefix):
        rep = rep_dataset_fn(x_train)
        out = {}
        for q in ["dynamic", "fp16", "int8"]:
            buf = tflite_from_keras(model, quant=q, rep_data=rep if q == "int8" else None, int8_io=True)
            path = EDGE_DIR / f"{name_prefix}_{q}.tflite"
            size_mb = write_tflite(buf, path)
            t_eval = eval_tflite(path, x_eval, y_eval, NUM_THREADS)
            out[q] = {
                "size_mb": size_mb,
                "latency_ms": t_eval["latency_ms_per_sample"],
                "accuracy": t_eval["tflite_accuracy"]
            }
        return out

    # student_base 量化
    results["quantization"]["student"] = quant_pack(student, "student")
    # 最优剪枝模型量化
    best_pruned_path = EDGE_DIR / f"pruned_{best_key}.keras"
    best_pruned = keras.models.load_model(best_pruned_path)
    results["quantization"][f"pruned_{best_key}"] = quant_pack(best_pruned, f"pruned_{best_key}")

    # —— 保存报告 —— #
    (REPORTS / "edge_prune_quant.json").write_text(json.dumps(results, indent=2))
    print("[done] wrote", REPORTS / "edge_prune_quant.json")

if __name__ == "__main__":
    main()
