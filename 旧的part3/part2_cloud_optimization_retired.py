
# part2_cloud_optimization_fixed.py
import os, time, json, argparse, pathlib, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

# -----------------------
# Global config / paths
# -----------------------
SEED = 42
VAL_RATIO = 0.1
ROOT = pathlib.Path(".").resolve()
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -----------------------
# Data pipeline helpers
# -----------------------
def _augment():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ],
        name="augment",
    )

def get_cifar10_datasets(batch_size=256, augment=True, out_dtype=tf.float32, place_aug_on_cpu=False):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    n = int(len(x_train) * (1.0 - VAL_RATIO))
    x_tr, y_tr = x_train[:n], y_train[:n]
    x_val, y_val = x_train[n:], y_train[n:]

    aug = _augment()

    def make(x, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
            opts = tf.data.Options(); opts.experimental_deterministic = False
            ds = ds.with_options(opts)
            if augment:
                if place_aug_on_cpu:
                    # ✅ 增强在 CPU 上以 fp32 执行，然后再 cast 到目标 dtype
                    def aug_map(a, b):
                        with tf.device("/CPU:0"):
                            a = aug(a, training=True)
                        return tf.cast(a, out_dtype), b
                    ds = ds.map(aug_map, num_parallel_calls=tf.data.AUTOTUNE)
                else:
                    ds = ds.map(lambda a, b: (aug(a, training=True), b),
                                num_parallel_calls=tf.data.AUTOTUNE)
        # 若上面没在 CPU 分支里 cast，这里再统一 cast 一次
        if not (training and augment and place_aug_on_cpu):
            ds = ds.map(lambda a, b: (tf.cast(a, out_dtype), b),
                        num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    return make(x_tr, y_tr, True), make(x_val, y_val, False), make(x_test, y_test, False), (x_tr, y_tr, x_val, y_val, x_test, y_test)



def get_cifar10_datasets_mp_noaug(batch_size=256):
    """Mixed precision 专用：无增强，float16 输入，避免 Metal 上的 f16/f32 冲突。"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = (x_train.astype("float16") / np.float16(255.0))
    x_test  = (x_test.astype("float16")  / np.float16(255.0))
    n = int(len(x_train) * (1.0 - VAL_RATIO))
    x_tr, y_tr = x_train[:n], y_train[:n]
    x_val, y_val = x_train[n:], y_train[n:]

    def make(x, y, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
            opts = tf.data.Options(); opts.experimental_deterministic = False
            ds = ds.with_options(opts)
        # no augmentation; already float16
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    return make(x_tr, y_tr, True), make(x_val, y_val, False), make(x_test, y_test, False)

# -----------------------
# Model builders
# -----------------------
def build_cnn(width_mult=1.0, dense_units=256):
    c1 = max(8, int(32 * width_mult))
    c2 = max(8, int(64 * width_mult))
    c3 = max(8, int(128 * width_mult))
    du = max(32, int(dense_units * width_mult))

    inputs = layers.Input(shape=(32, 32, 3))
    x = inputs

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
    outputs = layers.Dense(10)(x)  # logits
    return keras.Model(inputs, outputs)

def compile_for_classification(model, lr=1e-3, from_logits=True):
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])
    return model

# -----------------------
# Mixed Precision policy scope
# -----------------------
class PolicyScope:
    def __init__(self, policy_name):
        self.new = mixed_precision.Policy(policy_name)
        self.old = mixed_precision.global_policy()
    def __enter__(self):
        mixed_precision.set_global_policy(self.new)
    def __exit__(self, exc_type, exc, tb):
        mixed_precision.set_global_policy(self.old)

# -----------------------
# Gradient Accumulation (stable with Variables)
# -----------------------
class AccumModel(keras.Model):
    """Keras 模式下稳定的梯度累积实现。"""
    def __init__(self, inner, accum_steps=1):
        super().__init__()
        self.inner = inner
        self.accum_steps = tf.constant(int(accum_steps), dtype=tf.int32)
        self.accum_grads = [tf.Variable(tf.zeros_like(v), trainable=False) for v in inner.trainable_variables]
        self.step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def compile(self, optimizer, loss, metrics):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self.inner(x, training=True)
            loss = self.compiled_loss(y, logits, regularization_losses=self.inner.losses)
        grads = tape.gradient(loss, self.inner.trainable_variables)
        for acc, g, v in zip(self.accum_grads, grads, self.inner.trainable_variables):
            if g is None:
                g = tf.zeros_like(v)
            acc.assign_add(g)
        self.step.assign_add(1)

        def _apply():
            self.optimizer.apply_gradients(zip(self.accum_grads, self.inner.trainable_variables))
            for acc in self.accum_grads:
                acc.assign(tf.zeros_like(acc))
            return 0

        tf.cond(tf.equal(self.step % self.accum_steps, 0), _apply, lambda: 0)

        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        logits = self.inner(x, training=False)
        self.compiled_loss(y, logits, regularization_losses=self.inner.losses)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        return self.inner(inputs, training=training)

# -----------------------
# Knowledge Distillation
# -----------------------
class Distiller(keras.Model):
    def __init__(self, student, teacher, temperature=2.0, alpha=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.distill_loss_fn = keras.losses.KLDivergence()

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)

    def train_step(self, data):
        x, y = data
        teacher_logits = self.teacher(x, training=False)
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            s_loss = self.student_loss_fn(y, student_logits)
            t = self.temperature
            teacher_probs = tf.nn.softmax(teacher_logits / t, axis=-1)
            student_probs = tf.nn.softmax(student_logits / t, axis=-1)
            d_loss = self.distill_loss_fn(teacher_probs, student_probs) * (t * t)
            total = self.alpha * s_loss + (1.0 - self.alpha) * d_loss
        grads = tape.gradient(total, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.compiled_metrics.update_state(y, student_logits)
        return {"loss": total, **{m.name: m.result() for m in self.metrics}}

    def test_step(self, data):
        x, y = data
        student_logits = self.student(x, training=False)
        self.compiled_metrics.update_state(y, student_logits)
        return {m.name: m.result() for m in self.metrics}

# -----------------------
# Cloud Optimizer
# -----------------------
class CloudOptimizer:
    def __init__(self, baseline_model_path="baseline_model.keras"):
        self.baseline = tf.keras.models.load_model(baseline_model_path)

    # 1) Mixed Precision
    def implement_mixed_precision(self):
        with PolicyScope("mixed_float16"):
            model = build_cnn(width_mult=1.0, dense_units=256)
            # logits -> float32，避免损失/指标 dtype 冲突
            x = layers.Activation("linear", dtype="float32", name="logits_fp32")(model.outputs[0])
            model = keras.Model(model.inputs, x)
            compile_for_classification(model, lr=1e-3, from_logits=True)
        return model

    # 2) Distributed（本机模拟）
    def implement_model_parallelism(self, strategy="mirrored"):
        if strategy == "mirrored":
            strat = tf.distribute.MirroredStrategy()
        elif strategy == "multi_worker_mirrored":
            strat = tf.distribute.MultiWorkerMirroredStrategy()
        else:
            strat = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        with strat.scope():
            model = build_cnn(width_mult=1.0, dense_units=256)
            compile_for_classification(model, lr=1e-3, from_logits=True)
        return model, strat

    # 3) Batch processing（梯度累积 + 高吞吐数据管道）
    def optimize_batch_processing(self, target_batch_size=1024, micro_batch_size=256):
        accum_steps = max(1, target_batch_size // micro_batch_size)
        return {
            "target_batch_size": int(target_batch_size),
            "micro_batch_size": int(micro_batch_size),
            "accum_steps": int(accum_steps),
            "data_pipeline": {
                "prefetch": "tf.data.AUTOTUNE",
                "parallel_map": True,
                "non_deterministic": True,
                "shuffle_buffer": 8192
            }
        }

    # 4) Knowledge Distillation（≈2x 参数 teacher）
    def implement_knowledge_distillation(self, approx_multiplier=2.0):
        base_params = self.baseline.count_params()
        candidates = [1.25, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
        teacher, best_ratio = None, 0
        for w in candidates:
            m = build_cnn(width_mult=w, dense_units=int(256*w))
            p = m.count_params()
            ratio = p / base_params
            if ratio >= approx_multiplier and (best_ratio == 0 or ratio < best_ratio):
                teacher, best_ratio = m, ratio
        if teacher is None:
            teacher = build_cnn(width_mult=2.0, dense_units=512)
        student = build_cnn(width_mult=1.0, dense_units=256)
        return teacher, student

# -----------------------
# Benchmark helpers
# -----------------------
def train_and_time(model, train_ds, val_ds, epochs=3, callbacks=None):
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2, callbacks=callbacks)
    t1 = time.perf_counter()
    wall_s = t1 - t0
    per_epoch = wall_s / epochs
    return wall_s, per_epoch, hist  # 返回 History 对象

def evaluate(model, test_ds):
    # 兼容子类模型的 evaluate：用 return_dict=True
    out = model.evaluate(test_ds, verbose=0, return_dict=True)
    if isinstance(out, dict):
        loss = out.get("loss", list(out.values())[0])
        # pick the first key that looks like accuracy
        acc = None
        for k, v in out.items():
            if "acc" in k:
                acc = v
                break
        if acc is None:
            acc = list(out.values())[-1]
    else:
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            loss, acc = out[0], out[1]
        else:
            loss, acc = out, float("nan")
    return {"test_loss": float(loss), "test_accuracy": float(acc)}

# -----------------------
# Main benchmark
# -----------------------
def benchmark_cloud_optimizations(baseline_model_path="baseline_model.keras", quick=False, strategy="mirrored"):
    optimizer = CloudOptimizer(baseline_model_path)
    results = {}

    # 公共数据集
    base_batch = 256 if quick else 512
    train_ds, val_ds, test_ds, _ = get_cifar10_datasets(batch_size=base_batch, augment=True, out_dtype=tf.float32,
    place_aug_on_cpu=True )

    # Mixed Precision
    try:
        mp_model = optimizer.implement_mixed_precision()
        train_ds_mp, val_ds_mp, test_ds_mp = get_cifar10_datasets_mp_noaug(batch_size=base_batch)
        print("MP policy:", mixed_precision.global_policy())
        print("MP ds dtype:", next(iter(train_ds_mp))[0].dtype)
        ep = 2 if quick else 6
        mp_time_s, mp_epoch_s, _ = train_and_time(mp_model, train_ds_mp, val_ds_mp, epochs=ep)
        mp_eval = evaluate(mp_model, test_ds_mp)
        results["mixed_precision"] = {
            "epochs": ep,
            "total_time_s": round(mp_time_s, 2),
            "time_per_epoch_s": round(mp_epoch_s, 2),
            **mp_eval
        }
    except Exception as e:
        results["mixed_precision"] = {"error": str(e)}

    # Distributed / Parallelism
    try:
        dist_model, strat = optimizer.implement_model_parallelism(strategy=strategy)
        ep = 1 if quick else 3
        d_time_s, d_epoch_s, _ = train_and_time(dist_model, train_ds, val_ds, epochs=ep)
        d_eval = evaluate(dist_model, test_ds)
        results["distributed"] = {
            "strategy": type(strat).__name__,
            "replicas": getattr(strat, "num_replicas_in_sync", 1),
            "epochs": ep,
            "total_time_s": round(d_time_s, 2),
            "time_per_epoch_s": round(d_epoch_s, 2),
            **d_eval
        }
    except Exception as e:
        results["distributed"] = {"error": str(e)}

    # Batch processing / Gradient Accumulation
    try:
        cfg = optimizer.optimize_batch_processing(target_batch_size=1024 if quick else 2048, micro_batch_size=base_batch)
        base_model = build_cnn(width_mult=1.0, dense_units=256)
        compile_for_classification(base_model, lr=1e-3, from_logits=True)
        accum_model = AccumModel(base_model, accum_steps=cfg["accum_steps"])
        accum_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=["accuracy"])
        ep = 1 if quick else 3
        a_time_s, a_epoch_s, _ = train_and_time(accum_model, train_ds, val_ds, epochs=ep)
        a_eval = evaluate(accum_model, test_ds)

        # imgs/sec for a few batch sizes
        def epoch_imgs_per_sec(batch_size):
            ds_bs, _, _, _ = get_cifar10_datasets(batch_size=batch_size, augment=True)
            m = build_cnn()
            compile_for_classification(m, lr=1e-3, from_logits=True)
            t0 = time.perf_counter()
            hist = m.fit(ds_bs, epochs=1, verbose=0)
            t1 = time.perf_counter()
            steps = hist.params.get("steps", None) or 0
            elapsed = t1 - t0
            return round((steps * batch_size) / elapsed, 2) if elapsed > 0 and steps > 0 else None

        bs_list = [256, 512] if quick else [128, 256, 512, 1024]
        throughput = {str(bs): epoch_imgs_per_sec(bs) for bs in bs_list}

        results["batch_processing"] = {
            "config": cfg,
            "epochs": ep,
            "total_time_s": round(a_time_s, 2),
            "time_per_epoch_s": round(a_epoch_s, 2),
            **a_eval,
            "imgs_per_sec": throughput,
        }
    except Exception as e:
        results["batch_processing"] = {"error": str(e)}

    # Knowledge Distillation
    try:
        teacher, student = optimizer.implement_knowledge_distillation(approx_multiplier=2.0)
        ep_t = 4 if quick else 12
        teacher = compile_for_classification(teacher, lr=1e-3, from_logits=True)
        t_train, _, _ = train_and_time(teacher, train_ds, val_ds, epochs=ep_t)
        t_eval = evaluate(teacher, test_ds)

        ep_s = 4 if quick else 12
        student = compile_for_classification(student, lr=1e-3, from_logits=True)
        distiller = Distiller(student=student, teacher=teacher, temperature=2.0, alpha=0.5)
        distiller.compile(optimizer=keras.optimizers.Adam(1e-3), metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
        d_train, _, _ = train_and_time(distiller, train_ds, val_ds, epochs=ep_s)
        s_eval = evaluate(student, test_ds)

        results["knowledge_distillation"] = {
            "teacher_params": int(teacher.count_params()),
            "student_params": int(student.count_params()),
            "teacher_train_time_s": round(t_train, 2),
            "teacher_accuracy": t_eval["test_accuracy"],
            "student_train_time_s": round(d_train, 2),
            "student_accuracy": s_eval["test_accuracy"],
        }
    except Exception as e:
        results["knowledge_distillation"] = {"error": str(e)}

    out = REPORTS / "cloud_optimization_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    return results

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="baseline_model.keras")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--strategy", type=str, default="mirrored",
                        choices=["mirrored", "multi_worker_mirrored", "one_device"])
    args = parser.parse_args()

    results = benchmark_cloud_optimizations(
        baseline_model_path=args.baseline,
        quick=args.quick,
        strategy=args.strategy if args.strategy != "one_device" else "one_device",
    )
    print("Cloud Optimization Results:")
    for k, v in results.items():
        print(f"- {k}: {v}")
    print(f"Saved to {REPORTS / 'cloud_optimization_results.json'}")

if __name__ == "__main__":
    main()
