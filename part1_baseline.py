# part1_baseline.py
import os, json, time, random, pathlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -----------------------
# Global config
# -----------------------
SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
VAL_RATIO = 0.1

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Output dirs
ROOT = pathlib.Path(".").resolve()
(CHARTS := ROOT / "charts").mkdir(exist_ok=True, parents=True)
(REPORTS := ROOT / "reports").mkdir(exist_ok=True, parents=True)

def create_baseline_model():
    """
    Create a moderately complex CNN for CIFAR-10 classification.
    Intentionally over-parameterized as the baseline for later optimization.
    """
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),

        # Block 1
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.MaxPooling2D(),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_and_preprocess_data():
    """
    Load CIFAR-10 and normalize to [0,1].
    Return (x_train, y_train, x_test, y_test) as np arrays.
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    y_train = y_train.astype("int32")
    y_test  = y_test.astype("int32")
    return x_train, y_train, x_test, y_test


# Simple, fast augmentor (applied on-the-fly to training set only)
_AUG = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
], name="augment")


def _make_ds(x, y, batch=BATCH_SIZE, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        # non-deterministic map for speed
        opts = tf.data.Options()
        opts.experimental_deterministic = False
        ds = ds.with_options(opts)
        ds = ds.map(lambda a, b: (_AUG(a, training=True), b),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


def train_baseline_model(model, x_train, y_train, x_test, y_test):
    """
    Train with EarlyStopping/ReduceLROnPlateau/ModelCheckpoint, max 50 epochs.
    Returns: (best_model, history, metrics_dict)
    """
    # Train/Val split
    n = int(len(x_train) * (1.0 - VAL_RATIO))
    x_tr, y_tr = x_train[:n], y_train[:n]
    x_val, y_val = x_train[n:], y_train[n:]

    train_ds = _make_ds(x_tr, y_tr, training=True)
    val_ds   = _make_ds(x_val, y_val, training=False)
    test_ds  = _make_ds(x_test, y_test, training=False)

    ckpt_path = "baseline_model.keras"  # best-by-val_acc
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max",
            patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max",
            factor=0.5, patience=3, min_lr=1e-5, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=2,
        callbacks=callbacks,
    )

    # Plot training curve for deliverable “training history”
    fig = plt.figure(figsize=(6,4))
    acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
    plt.plot(acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    curve_path = CHARTS / "baseline_accuracy.png"
    fig.savefig(curve_path); plt.close(fig)

    # Save raw history for reproducibility
    with open(REPORTS / "baseline_history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    # Evaluate on test
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    # Inference latency (single image, averaged)
    @tf.function
    def _infer_one(b):
        return model(b, training=False)

    dummy = tf.zeros([1, 32, 32, 3], dtype=tf.float32)
    for _ in range(10):  # warmup
        _ = _infer_one(dummy)
    runs = 100
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = _infer_one(dummy)
    latency_ms = (time.perf_counter() - t0) / runs * 1000.0

    params = int(model.count_params())
    # memory footprint (parameters * 4 bytes) as an estimate
    param_mem_mb = params * 4 / (1024 ** 2)

    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "params": params,
        "parameter_memory_mb": float(param_mem_mb),
        "inference_time_ms": float(latency_ms),
        "history_curve_png": str(curve_path),
    }
    # Save metrics (without model artifact size yet)
    with open(REPORTS / "baseline_metrics_partial.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, history.history, metrics


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Create & train
    model = create_baseline_model()
    model, history, metrics = train_baseline_model(model, x_train, y_train, x_test, y_test)

    # Save baseline model (final copy; best weights already saved by checkpoint)
    model.save("baseline_model.keras", overwrite=True)

    # Compute model file size for deliverable
    size_mb = os.path.getsize("baseline_model.keras") / (1024 ** 2)
    metrics["model_file_mb"] = float(size_mb)

    # Persist final metrics
    with open(REPORTS / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline model parameters: {model.count_params():,}")
    print(f"Baseline test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Inference time (1x32x32x3): {metrics['inference_time_ms']:.2f} ms")
    print(f"Model file size: {metrics['model_file_mb']:.2f} MB")
    print(f"Training curve saved to: {metrics['history_curve_png']}")
