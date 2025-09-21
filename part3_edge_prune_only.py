# part3_edge_prune_only.py
# 环境: conda env: mls-a2-old (TensorFlow 2.15.x + tfmot 0.8.0 + (可选) tf_keras 2.15.x)
# 仅此脚本需要 legacy keras：
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # 放最顶部

import json, math, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- Config ----------------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

ROOT = Path(".").resolve()
EDGE_DIR = ROOT / "edge_optimized_models"; EDGE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS = ROOT / "reports"; REPORTS.mkdir(parents=True, exist_ok=True)
CHARTS = ROOT / "charts"; CHARTS.mkdir(parents=True, exist_ok=True)

STU_W = "cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.weights.h5"

BATCH = 32
EPOCHS_FINETUNE = 6           # 5–10 之间
SPARSITIES = [0.50, 0.75, 0.90]
TEST_SAMPLES = 1000           # 评测样本量
VAL_SPLIT = 0.1               # 微调时从训练划出验证

# ---------------- Data ----------------
def load_cifar10():
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32") / 255.0
    xte = xte.astype("float32") / 255.0
    return (xtr, ytr.flatten()), (xte, yte.flatten())

def take_subset(x, y, n):
    n = min(n, len(x))
    return x[:n], y[:n]

# ---------------- Model ----------------
def create_student_architecture():
    i = layers.Input((32,32,3))
    x = i
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    o = layers.Dense(10)(x)  # logits
    return keras.Model(i, o)

def clone_with_weights(model):
    m = keras.models.clone_model(model)
    m.set_weights(model.get_weights())
    return m

def compile_for_train(m, lr=1e-3):
    m.compile(
        optimizer=keras.optimizers.legacy.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
    return m

# ---------------- Pruning ----------------
def prune_and_finetune(base_model, xtr, ytr, sparsity, epochs, batch):
    # 计算剪枝步数：使用训练步数作为 polynomial decay 基准
    steps_per_epoch = math.ceil(len(xtr) * (1.0-VAL_SPLIT) / batch)
    end_step = steps_per_epoch * epochs

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=float(sparsity),
            begin_step=0,
            end_step=end_step
        )
    }
    pruned = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    compile_for_train(pruned)

    cbs = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
    ]

    pruned.fit(
        xtr, ytr,
        batch_size=batch,
        epochs=epochs,
        validation_split=VAL_SPLIT,
        verbose=2,
        callbacks=cbs
    )

    # strip 剪枝包装以便真正稀疏化权重
    stripped = tfmot.sparsity.keras.strip_pruning(pruned)
    return stripped, pruned  # 返回 stripped 与 包装版（可用于统计掩码）

def measure_sparsity_approx(model):
    # 统计 Conv / Dense 的零比率（strip 后为真实零）
    zeros, total = 0, 0
    for w in model.weights:
        # 仅统计 kernel（跳过 BN 等权重）
        if any(k in w.name for k in ["kernel:0", "depthwise_kernel:0", "pointwise_kernel:0"]):
            v = w.numpy()
            total += v.size
            zeros += np.count_nonzero(v == 0)
    return (zeros / total) if total > 0 else None

def evaluate_keras(m, x, y):
    out = m.evaluate(x, y, batch_size=BATCH, verbose=0, return_dict=True)
    acc = float(next((out[k] for k in out if "acc" in k), list(out.values())[-1]))
    return {"loss": float(out.get("loss", list(out.values())[0])), "accuracy": acc}

# ---------------- Main ----------------
def main():
    (xtr, ytr), (xte, yte) = load_cifar10()
    x_eval, y_eval = take_subset(xte, yte, TEST_SAMPLES)

    # 基线：按你提供的 student 定义 + 加载 Part2 的权重(.h5)
    base = create_student_architecture()
    base.load_weights(STU_W)
    compile_for_train(base)
    base_eval = evaluate_keras(base, x_eval, y_eval)

    results = {
        "config": {
            "epochs_finetune": EPOCHS_FINETUNE,
            "batch": BATCH,
            "val_split": VAL_SPLIT,
            "test_samples": TEST_SAMPLES,
            "sparsities": SPARSITIES
        },
        "baseline_eval": base_eval,
        "pruning": []
    }

    # 逐个稀疏率执行剪枝 + 微调 + 评估 + 保存
    for sp in SPARSITIES:
        start = time.perf_counter()
        work = clone_with_weights(base)
        stripped, wrapped = prune_and_finetune(work, xtr, ytr, sp, EPOCHS_FINETUNE, BATCH)
        t = time.perf_counter() - start
        compile_for_train(stripped)  # 为 stripped 模型添加编译步骤
        eval_res = evaluate_keras(stripped, x_eval, y_eval)
        sparsity_measured = measure_sparsity_approx(stripped)

        # 保存 .keras
        tag = f"s{str(sp).replace('.','_')}"
        out_path = EDGE_DIR / f"student_pruned_{tag}.keras"
        stripped.save(out_path)

        results["pruning"].append({
            "target_sparsity": float(sp),
            "measured_sparsity": sparsity_measured,
            "eval": eval_res,
            "finetune_seconds": t,
            "keras_path": str(out_path)
        })
        print(f"[OK] sparsity={sp:.2f}  acc={eval_res['accuracy']:.4f}  saved: {out_path.name}")

    # 写报告
    rpt_path = REPORTS / "edge_pruning_results.json"
    rpt_path.write_text(json.dumps(results, indent=2))
    print("[done] wrote", rpt_path)

if __name__ == "__main__":
    main()
