import os, time, argparse, numpy as np, tensorflow as tf
parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True)
parser.add_argument("--n", type=int, default=5000); args = parser.parse_args()

def pin_cpu_only():
    try: tf.config.set_visible_devices([], "GPU"); tf.config.set_visible_devices([], "TPU")
    except: pass
pin_cpu_only()

(xtr, ytr), (x, y) = tf.keras.datasets.cifar10.load_data()
x = x[:args.n].astype("float32")/255.0; y = y[:args.n].reshape(-1)
m = tf.keras.models.load_model(args.model)

# accuracy
p = m.predict(x, batch_size=64, verbose=0); acc = (np.argmax(p,1)==y).mean()

# latency (single-sample, CPU)
sample = x[:1]
for _ in range(10): m.predict(sample, batch_size=1, verbose=0)
t0=time.perf_counter(); 
for _ in range(100): m.predict(sample, batch_size=1, verbose=0)
lat=(time.perf_counter()-t0)/100*1000
print(f"ACC={acc:.4f}  LAT(ms)={lat:.4f}")
