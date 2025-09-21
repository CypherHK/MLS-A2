# export_weights_from_keras.py  (run in: mls-a2)
import argparse
from pathlib import Path
import keras

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--keras", required=True, help="path to .keras")
    ap.add_argument("--out",   required=True, help="path to .weights.h5")
    args=ap.parse_args()
    m=keras.models.load_model(args.keras)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.save_weights(args.out)
    print("[done] wrote", args.out)

if __name__=="__main__":
    main()
