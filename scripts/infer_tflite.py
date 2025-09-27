#!/usr/bin/env python3
# Robust TFLite inference (Edge/Tiny). Prints progress and flushes output.
import os, sys, time, argparse, numpy as np, tensorflow as tf

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_cifar10_test(n=None):
    (xtr, ytr), (x, y) = tf.keras.datasets.cifar10.load_data()
    x = x.astype("float32")/255.0
    y = y.reshape(-1)
    if n is not None:
        x, y = x[:n], y[:n]
    return x, y

def quantize_like(interp, x):
    d = interp.get_input_details()[0]
    scale, zero = d["quantization"]
    dt = d["dtype"]
    if scale == 0:
        return x.astype(dt)
    q = np.round(x/scale + zero).astype(dt)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        q = np.clip(q, info.min, info.max)
    return q

def main():
    ap = argparse.ArgumentParser(description="TFLite inference (accuracy & latency)")
    ap.add_argument("--model", required=True, help="Path to .tflite")
    ap.add_argument("--n", type=int, default=5000, help="Number of test samples")
    ap.add_argument("--runs", type=int, default=100, help="Latency averaging runs")
    ap.add_argument("--warmup", type=int, default=20, help="Latency warmup runs")
    ap.add_argument("--num_threads", type=int, default=1, help="TFLite threads")
    args = ap.parse_args()

    print(f"[INFO] model={args.model}", flush=True)
    if not os.path.exists(args.model):
        eprint(f"[ERROR] File not found: {args.model}")
        sys.exit(2)

    # Ensure CPU-only for stable latency
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
    except Exception:
        pass

    print("[INFO] Loading CIFAR-10 test set...", flush=True)
    x, y = load_cifar10_test(args.n)
    print(f"[INFO] Test subset: {len(x)} samples", flush=True)

    print("[INFO] Loading TFLite interpreter...", flush=True)
    interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=args.num_threads)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print("[INFO] Input tensor:", inp, flush=True)

    # latency
    sample = quantize_like(interpreter, x[:1])
    for _ in range(args.warmup):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
    t0 = time.perf_counter()
    for _ in range(args.runs):
        interpreter.set_tensor(inp["index"], sample)
        interpreter.invoke()
    t1 = time.perf_counter()
    lat_ms = (t1 - t0)/args.runs*1000.0
    print(f"[INFO] Latency(ms) ~ {lat_ms:.6f}", flush=True)

    # accuracy (first 2000)
    N = min(2000, len(x))
    correct = 0
    for i in range(N):
        xi = quantize_like(interpreter, x[i:i+1])
        interpreter.set_tensor(inp["index"], xi)
        interpreter.invoke()
        logits = interpreter.get_tensor(out["index"])
        pred = int(np.argmax(logits, axis=1)[0])
        if pred == int(y[i]):
            correct += 1
        if (i+1) % 500 == 0:
            print(f"[INFO] Eval progress: {i+1}/{N}", flush=True)
    acc = correct/N
    print(f"[RESULT] ACC={acc:.4f}  LAT(ms)={lat_ms:.6f}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
