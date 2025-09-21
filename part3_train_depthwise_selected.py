# train_depthwise_selected.py  (run in: mls-a2)
import argparse, json
from pathlib import Path
import numpy as np, tensorflow as tf, keras
from keras import layers
keras.mixed_precision.set_global_policy("float32")

ROOT=Path(".").resolve()
EDGE=ROOT/"edge_optimized_models"; EDGE.mkdir(exist_ok=True, parents=True)
REPORTS=ROOT/"reports"; REPORTS.mkdir(exist_ok=True, parents=True)
SEED=42; np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)

def build_depthwise_student(width_mult=0.75, depth_mult=1.0, dense_units=192):
    c1, c2, c3 = [max(4, int(v * width_mult)) for v in (32, 64, 128)]
    rep = lambda: max(1, int(2 * depth_mult))
    rep1 = rep(); rep2 = rep(); rep3 = rep()
    du = max(16, int(dense_units * width_mult))

    i = layers.Input((32,32,3))
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
    x=layers.Dense( max(16,int(dense_units*width_mult)), activation="relu")(x)
    x=layers.Dropout(0.25)(x)
    o=layers.Dense(10)(x)
    return keras.Model(i,o)

def compile_model(m, lr=1e-3):
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=0.75)
    ap.add_argument("--depth", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    args=ap.parse_args()

    (xtr,ytr),(xte,yte)=keras.datasets.cifar10.load_data()
    xtr=xtr.astype("float32")/255.; xte=xte.astype("float32")/255.
    ytr=ytr.flatten(); yte=yte.flatten()
    n=int(0.9*len(xtr)); x_train,y_train=xtr[:n],ytr[:n]; x_val,y_val=xtr[n:],ytr[n:]

    m=build_depthwise_student(args.width,args.depth)
    compile_model(m)
    cbs=[keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True),
         keras.callbacks.ReduceLROnPlateau(patience=2,factor=0.5,min_lr=1e-5)]
    m.fit(x_train,y_train,validation_data=(x_val,y_val),
          epochs=args.epochs,batch_size=args.batch,verbose=2,callbacks=cbs)
    ev=m.evaluate(xte,yte,batch_size=args.batch,verbose=0,return_dict=True)
    acc=float(ev.get("accuracy", list(ev.values())[-1]))

    tag=f"depthwise_w{str(args.width).replace('.','_')}_d{str(args.depth).replace('.','_')}"
    kp=EDGE/f"{tag}.keras"; wp=EDGE/f"{tag}.weights.h5"
    m.save(kp); m.save_weights(wp)
    out={"width":args.width,"depth":args.depth,"epochs":args.epochs,
         "keras_path":str(kp),"weights_path":str(wp),"test_accuracy":acc}
    (REPORTS/"train_depthwise_selected.json").write_text(json.dumps(out,indent=2))
    print("[done] wrote", REPORTS/"train_depthwise_selected.json")

if __name__ == "__main__":
    main()
