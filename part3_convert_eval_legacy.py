# convert_eval_legacy.py  (run in: mls-a2-old  TF 2.15 + tf_keras 2.15 + macOS)
import os; os.environ["TF_USE_LEGACY_KERAS"]="1"
import argparse, json, time
from pathlib import Path
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers

SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED)
ROOT=Path(".").resolve()
EDGE=ROOT/"edge_optimized_models"; EDGE.mkdir(parents=True, exist_ok=True)
REPORTS=ROOT/"reports"; REPORTS.mkdir(parents=True, exist_ok=True)

BATCH=32; TEST_SAMPLES=1000; REP_SAMPLES=400
NUM_THREADS=1; WARMUP=10; RUNS=200

def build_student():
    i = layers.Input((32,32,3))
    x = i
    for f in [32,64,128]:
        x=layers.Conv2D(f,3,padding="same",use_bias=False)(x)
        x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
        x=layers.Conv2D(f,3,padding="same",use_bias=False)(x)
        x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
        x=layers.MaxPooling2D()(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(0.5)(x)
    x=layers.Dense(256,activation="relu")(x)
    x=layers.Dropout(0.3)(x)
    o=layers.Dense(10)(x)
    return keras.Model(i,o)

def build_depthwise(width_mult=0.75, depth_mult=1.0, dense_units=192):
    c1,c2,c3=[max(4,int(v*width_mult)) for v in (32,64,128)]
    rep=lambda: max(1,int(2*depth_mult))
    rep1,rep2,rep3=rep(),rep(),rep()
    du=max(16,int(dense_units*width_mult))

    i=layers.Input((32,32,3))
    x=i
    for _ in range(rep1):
        x=layers.DepthwiseConv2D(3,padding="same",use_bias=False)(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
        x=layers.Conv2D(c1,1,padding="same",use_bias=False)(x);      x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.MaxPooling2D()(x)
    for _ in range(rep2):
        x=layers.DepthwiseConv2D(3,padding="same",use_bias=False)(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
        x=layers.Conv2D(c2,1,padding="same",use_bias=False)(x);      x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.MaxPooling2D()(x)
    for _ in range(rep3):
        x=layers.DepthwiseConv2D(3,padding="same",use_bias=False)(x); x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
        x=layers.Conv2D(c3,1,padding="same",use_bias=False)(x);      x=layers.BatchNormalization()(x); x=layers.ReLU()(x)
    x=layers.MaxPooling2D()(x)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(0.4)(x)
    x=layers.Dense(du,activation="relu")(x)
    x=layers.Dropout(0.25)(x)
    o=layers.Dense(10)(x)
    return keras.Model(i,o)

def load_cifar10():
    (xtr,ytr),(xte,yte)=keras.datasets.cifar10.load_data()
    xtr=xtr.astype("float32")/255.; xte=xte.astype("float32")/255.
    return (xtr,ytr.flatten()), (xte,yte.flatten())

def rep_dataset(xtr, n=REP_SAMPLES):
    idx=np.random.RandomState(SEED).choice(len(xtr), size=min(n,len(xtr)), replace=False)
    for i in idx:
        yield [xtr[i:i+1].astype("float32")]

def write_tflite(buf, path:Path):
    path.write_bytes(buf); return path.stat().st_size/1e6

def _adapt_input(x1, inp_detail):
    dtype=inp_detail["dtype"]
    if dtype==np.float32: return x1.astype(np.float32)
    scale,zp=inp_detail.get("quantization", (0.0,0))
    if not scale: return x1.astype(dtype)
    q=np.round(x1/scale + zp)
    q=np.clip(q, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return q

def eval_tflite(path:Path, x, y, threads=NUM_THREADS, warmup=WARMUP, runs=RUNS):
    itp=tf.lite.Interpreter(model_path=str(path), num_threads=threads); itp.allocate_tensors()
    inp=itp.get_input_details()[0]; out=itp.get_output_details()[0]
    dummy=np.zeros(inp["shape"], dtype=inp["dtype"])
    for _ in range(warmup): itp.set_tensor(inp["index"], dummy); itp.invoke()
    n=min(runs,len(x)); ok=0; t0=time.perf_counter()
    for i in range(n):
        xi=_adapt_input(x[i:i+1].astype("float32"), inp)
        itp.set_tensor(inp["index"], xi); itp.invoke()
        pred=int(itp.get_tensor(out["index"]).argmax(-1)); ok+=int(pred==int(y[i]))
    dt=time.perf_counter()-t0
    return {"latency_ms_per_sample": (dt/n)*1000, "accuracy": ok/n, "samples": n}

def convert_all(model, tag, xtr, x_eval, y_eval, do_int8=True, tflm_full_int8=True):
    out={}
    # Dynamic (FP32 weights quant)
    conv=tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]
    buf=conv.convert()
    p=EDGE/f"{tag}_drq.tflite"; out["drq"]={"size_mb":write_tflite(buf,p), **eval_tflite(p,x_eval,y_eval)}

    # FP16
    conv=tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations=[tf.lite.Optimize.DEFAULT]; conv.target_spec.supported_types=[tf.float16]
    buf=conv.convert()
    p=EDGE/f"{tag}_fp16.tflite"; out["fp16"]={"size_mb":write_tflite(buf,p), **eval_tflite(p,x_eval,y_eval)}

    # INT8
    if do_int8:
        conv=tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations=[tf.lite.Optimize.DEFAULT]
        conv.representative_dataset=lambda: rep_dataset(xtr, REP_SAMPLES)
        if tflm_full_int8:
            conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type=tf.int8; conv.inference_output_type=tf.int8
        buf=conv.convert()
        p=EDGE/f"{tag}_int8.tflite"; out["int8"]={"size_mb":write_tflite(buf,p), **eval_tflite(p,x_eval,y_eval)}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["student","depthwise"], required=True)
    ap.add_argument("--width", type=float, default=0.75)   # only for depthwise
    ap.add_argument("--depth", type=float, default=1.0)    # only for depthwise
    ap.add_argument("--weights", required=True, help="path to .weights.h5")
    ap.add_argument("--int8", action="store_true", help="also export INT8 (full integer if possible)")
    args=ap.parse_args()

    (xtr,_),(xte,yte)=load_cifar10()
    x_eval, y_eval = xte[:TEST_SAMPLES], yte[:TEST_SAMPLES]

    if args.arch=="student":
        m=build_student()
        tag="student"
    else:
        m=build_depthwise(args.width, args.depth)
        tag=f"depthwise_w{str(args.width).replace('.','_')}_d{str(args.depth).replace('.','_')}"
    m.load_weights(args.weights)

    # 转换与评测
    packs=convert_all(m, tag, xtr, x_eval, y_eval, do_int8=args.int8, tflm_full_int8=True)

    res={"arch":args.arch, "width":args.width, "depth":args.depth,
         "weights_path":args.weights, "tflite":packs}
    (REPORTS/"edge_quant_results_legacy.json").write_text(json.dumps(res,indent=2))
    print("[done] wrote", REPORTS/"edge_quant_results_legacy.json")

if __name__=="__main__":
    main()
