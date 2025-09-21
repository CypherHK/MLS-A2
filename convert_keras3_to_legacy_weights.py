# convert_keras3_to_legacy_weights.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"  # 强制使用 Keras 3 来读取 .keras

import tensorflow as tf
import keras
from pathlib import Path

SRC = "cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.keras"
DST = "cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.weights.h5"

m = keras.saving.load_model(SRC)  # Keras 3 的加载接口
Path(DST).parent.mkdir(parents=True, exist_ok=True)
m.save_weights(DST)               # 保存成 legacy 可读的权重 .h5
print("[done] wrote", DST)
