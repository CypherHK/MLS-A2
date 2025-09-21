# part3_edge_prune_only.py
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # 仅此脚本需要；放最顶部

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers

STU_W = "cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.weights.h5"

def create_student_arch():
    i = layers.Input((32,32,3))
    x = i
    for f in [32,64,128]:
        x = layers.Conv2D(f,3,padding="same",use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(f,3,padding="same",use_bias=False)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    o = layers.Dense(10)(x)
    return keras.Model(i,o)

# 重建 + 加载权重
base = create_student_arch()
base.load_weights(STU_W)
# 之后：tfmot 剪枝 + 微调 + strip_pruning + 保存 *.weights.h5 / *.tflite（与之前逻辑一致）
