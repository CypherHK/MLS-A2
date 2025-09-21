# part2_cloud_rest_only.py
import os, json, time, pathlib, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

def _augment():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ],
        name="augment",
    )

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

def get_ds_cpu_aug(bs=256):
    (xtr, ytr), (xte, yte) = keras.datasets.cifar10.load_data()
    xtr = xtr.astype("float32") / 255.0
    xte = xte.astype("float32") / 255.0
    n = int(len(xtr) * (1 - VAL_RATIO))
    x1, y1 = xtr[:n], ytr[:n]
    xval, yval = xtr[n:], ytr[n:]
    aug = _augment()

    def make(x, y, train=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if train:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
            def cpu_aug(a, b):
                with tf.device("/CPU:0"):
                    a = aug(a, training=True)
                return a, b
            ds = ds.map(cpu_aug, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(bs).prefetch(tf.data.AUTOTUNE)

    return make(x1, y1, True), make(xval, yval, False), make(xte, yte, False)

class AccumModel(keras.Model):
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

#def evaluate(model, ds):
   # out = model.evaluate(ds, verbose=0, return_dict=True)
   # return {"test_loss": float(out.get("loss", 0.0)), "test_accuracy": float(out.get("accuracy", 0.0))}

def _first_number(x):
    # 递归地从 x 中拿到第一个标量数值（float）
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if tf.is_tensor(x):
        try:
            return float(x.numpy())
        except Exception:
            pass
    if isinstance(x, dict):
        for v in x.values():
            r = _first_number(v)
            if r is not None:
                return r
    if isinstance(x, (list, tuple)):
        for v in x:
            r = _first_number(v)
            if r is not None:
                return r
    return None

def evaluate(model, ds):
    out = model.evaluate(ds, verbose=0, return_dict=True)
    # loss
    loss = _first_number(out.get("loss", out))
    # accuracy-like key（优先挑含 "acc" 的键）
    acc = None
    for k, v in out.items():
        if "acc" in k.lower():
            acc = _first_number(v)
            break
    if acc is None:
        # 兜底：拿最后一个数值
        acc = _first_number(list(out.values())[-1])
    return {
        "test_loss": float(loss) if loss is not None else float("nan"),
        "test_accuracy": float(acc) if acc is not None else float("nan"),
    }

def fit_and_time(model, train_ds, val_ds, epochs):
    t0 = time.perf_counter()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    t1 = time.perf_counter()
    total = t1 - t0
    return total, total / epochs, hist


def main():
    # distributed (local simulation)
    EP_D = 5
    tr, val, te = get_ds_cpu_aug()
    strat = tf.distribute.MirroredStrategy()
    with strat.scope():
        m = build_cnn()
        compile_model(m)

    #hist = m.fit(tr, validation_data=val, epochs=EP_D, verbose=2)
    
    total, per_epoch, hist = fit_and_time(m, tr, val, EP_D)
    (Path("reports")/"history_distributed.json").write_text(json.dumps(hist.history, indent=2))

    d_eval = evaluate(m, te)
    m.save(MODELS_DIR / f"distributed_fp32_ep{EP_D}.keras")
    distributed = {
        "strategy": "MirroredStrategy",
        "replicas": getattr(strat, "num_replicas_in_sync", 1),
        "epochs": EP_D,
        "total_time_s": round(total, 2),
        "time_per_epoch_s": round(per_epoch, 2),
        **d_eval,
    }

    # batch processing (gradient accumulation)
    EP_B = 10  # 比如 5–10
    base = build_cnn()
    compile_model(base)
    accum = AccumModel(base, accum_steps=4)
    accum.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    #hist = accum.fit(tr, validation_data=val, epochs=EP_B, verbose=2)
    
    total, per_epoch, hist = fit_and_time(accum, tr, val, EP_B)
    (Path("reports")/"history_batch_accum.json").write_text(json.dumps(hist.history, indent=2))

    a_eval = evaluate(accum, te)
    base.save(MODELS_DIR / f"batch_accum_fp32_ep{EP_B}.keras")
    batch = {
        "config": {"target_batch_size": 1024, "micro_batch_size": 256, "accum_steps": 4},
        "epochs": EP_B,
        "total_time_s": round(total, 2),
        "time_per_epoch_s": round(per_epoch, 2),
        **a_eval,
    }

    # knowledge distillation
    EP_T = 16  # 比如 12–20
    EP_S = 16
    ALPHA = 0.7
    TEMP = 2.0
    teacher = build_cnn(width_mult=1.6, dense_units=int(256 * 1.6))
    compile_model(teacher)
    t_total, t_per_epoch, hist_t = fit_and_time(teacher, tr, val, EP_T)
    # hist_t = teacher.fit(tr, validation_data=val, epochs=EP_T, verbose=2)
    (Path("reports")/"history_kd_teacher.json").write_text(json.dumps(hist_t.history, indent=2))
    t_eval = evaluate(teacher, te)
    teacher.save(MODELS_DIR / f"kd_teacher_ep{EP_T}.keras")

    student = build_cnn()
    compile_model(student)

    class Distiller(keras.Model):
        def __init__(self, student, teacher, temperature=2.0, alpha=0.5):
            super().__init__()
            self.s = student
            self.t = teacher
            self.T = temperature
            self.A = alpha
            self.ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self.kl = keras.losses.KLDivergence()

        def compile(self, optimizer, metrics):
            super().compile(optimizer=optimizer, metrics=metrics)

        def train_step(self, data):
            x, y = data
            tlog = self.t(x, training=False)
            with tf.GradientTape() as tape:
                slog = self.s(x, training=True)
                s_loss = self.ce(y, slog)
                t = tf.nn.softmax(tlog / self.T)
                s = tf.nn.softmax(slog / self.T)
                d = self.kl(t, s) * (self.T * self.T)
                tot = self.A * s_loss + (1 - self.A) * d
            g = tape.gradient(tot, self.s.trainable_variables)
            self.optimizer.apply_gradients(zip(g, self.s.trainable_variables))
            self.compiled_metrics.update_state(y, slog)
            return {"loss": tot, **{m.name: m.result() for m in self.metrics}}

        def test_step(self, data):
            x, y = data
            slog = self.s(x, training=False)
            self.compiled_metrics.update_state(y, slog)
            return {m.name: m.result() for m in self.metrics}

    dist = Distiller(student, teacher, 2.0, 0.5)
    dist.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    s_total, s_per_epoch, hist_s = fit_and_time(dist, tr, val, EP_S)
    (Path("reports")/"history_kd_student.json").write_text(json.dumps(hist_s.history, indent=2))

    s_eval = evaluate(student, te)
    
    def _fmt(x):  # 文件名里别带小数点太多
        return str(x).replace(".", "_")

    student.save(MODELS_DIR / f"kd_student_ep{EP_S}_alpha{_fmt(ALPHA)}_T{_fmt(TEMP)}.keras")
  
    kd = {
        "teacher_params": int(teacher.count_params()),
        "student_params": int(student.count_params()),
        "teacher_epochs": EP_T,
        "teacher_train_time_s": round(t_total, 2),
        "teacher_time_per_epoch_s": round(t_per_epoch, 2),
        "teacher_accuracy": t_eval["test_accuracy"],
        "student_epochs": EP_S,
        "student_train_time_s": round(s_total, 2),
        "student_time_per_epoch_s": round(s_per_epoch, 2),
        "student_accuracy": s_eval["test_accuracy"],
        "alpha": ALPHA,
        "temperature": TEMP,
    }
    
    out = {"distributed": distributed, "batch_processing": batch, "knowledge_distillation": kd}
    (ROOT / "reports" / "cloud_rest.json").write_text(json.dumps(out, indent=2))
    print(out)

if __name__ == "__main__":
    main()
