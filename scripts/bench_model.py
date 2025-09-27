import os, csv, time, argparse, numpy as np, tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--type", choices=["keras","tflite"], required=True)
parser.add_argument("--path", required=True)
parser.add_argument("--n", type=int, default=5000)
parser.add_argument("--runs", type=int, default=100)
parser.add_argument("--out", default="bench_results.csv")
args = parser.parse_args()

def pin_cpu_only():
    try: tf.config.set_visible_devices([], "GPU"); tf.config.set_visible_devices([], "TPU")
    except: pass
pin_cpu_only()

(xtr, ytr), (x, y) = tf.keras.datasets.cifar10.load_data()
x = x[:args.n].astype("float32")/255.0; y=y[:args.n].reshape(-1)

def eval_keras(path):
    m = tf.keras.models.load_model(path)
    p = m.predict(x, batch_size=64, verbose=0)
    acc = float((np.argmax(p,1)==y).mean())
    s = x[:1]
    for _ in range(10): m.predict(s, batch_size=1, verbose=0)
    t0=time.perf_counter()
    for _ in range(args.runs): m.predict(s, batch_size=1, verbose=0)
    lat=(time.perf_counter()-t0)/args.runs*1000
    return acc, lat

def q_like(interp, xb):
    d = interp.get_input_details()[0]; scale, zero = d["quantization"]; dt = d["dtype"]
    if scale == 0: return xb.astype(dt)
    q = np.round(xb/scale + zero).astype(dt)
    if np.issubdtype(dt, np.integer): 
        info = np.iinfo(dt); q = np.clip(q, info.min, info.max)
    return q

def eval_tflite(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
    s = q_like(interp, x[:1])
    for _ in range(20): interp.set_tensor(inp["index"], s); interp.invoke()
    t0=time.perf_counter()
    for _ in range(args.runs): interp.set_tensor(inp["index"], s); interp.invoke()
    lat=(time.perf_counter()-t0)/args.runs*1000
    N=min(2000, len(x)); c=0
    for i in range(N):
        xi=q_like(interp, x[i:i+1]); interp.set_tensor(inp["index"], xi); interp.invoke()
        logits=interp.get_tensor(out["index"]); c+= int(np.argmax(logits,1)[0]==int(y[i]))
    acc = c/N
    return acc, lat

acc, lat = eval_keras(args.path) if args.type=="keras" else eval_tflite(args.path)

size_mb = os.path.getsize(args.path)/1e6
mem_mb  = size_mb + 1.0 if args.type=="tflite" else size_mb*2 + (32*32*3*4)/1e6

with open(args.out, "a", newline="") as f:
    w=csv.writer(f)
    w.writerow([args.type, args.path, f"{size_mb:.6f}", f"{acc:.4f}", f"{lat:.6f}", f"{mem_mb:.6f}"])

print(f"type={args.type} path={args.path}")
print(f"size(MB)={size_mb:.6f}  acc={acc:.4f}  latency(ms)={lat:.6f}  mem(MB)={mem_mb:.6f}")
