设备与环境：
- Mac M1，conda env: mls-a2
- Python 3.10.x
- TensorFlow/Keras 版本：TF (2.16.x)，Keras (3.x)
- 当前未安装 tf_keras；未设置全局 TF_USE_LEGACY_KERAS
- 仅 CPU/Metal，禁止启用混合精度
-(当前环境包含了新版的tf与keras 3，稍后准备再另外建一个conda，用于旧版tf相关，因为TensorFlow Model Optimization与 Keras 3 就不兼容)
目录与模型：
- 这是当前项目的结构树：
MLS-A2/
  cloud_optimized_models/
    kd_student_ep16_alpha0_7_T2_0.keras               # Part 2 产物（Keras3）
    kd_student_ep16_alpha0_7_T2_0.weights.h5          # 我已转换好的权重（给 Part 3 用）
  edge_optimized_models/                              # Part 3 产物将写到这里
  reports/
    cloud_optimization_results.json                   # （如有）Part 2 汇总
  charts/
  part3_edge_prune_quant.py                           # 你将粘贴/生成
  part3_edge_arch_nas.py                              # 你将粘贴/生成
  part3_edge_plot.py                                  # 你将粘贴/生成

- KD 学生模型（Keras3 保存）已转换为权重：cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.weights.h5
- 学生架构定义如下（用于 load_weights）：
  # student architecture (CIFAR-10)
import keras
from keras import layers

def create_student_architecture():
    i = layers.Input((32,32,3))
    x = i
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    o = layers.Dense(10)(x)  # logits
    return keras.Model(i, o)


老师邮件（Part 3 指南）与示范代码前文已经提到，这里不再重复。

我的选择与参数（你觉得更好可以质疑）：
- 剪枝路线：使用 tf_keras + tfmot（仅在剪枝脚本中设置 TF_USE_LEGACY_KERAS=1）
- 训练预算：剪枝微调 5–10 epoch，Depthwise-Student 10–15 epoch，NAS 每候选 3–5 epoch
- 量化：Dynamic/FP16/INT8（rep=400），TFLite 评测 TEST_SAMPLES=1000，num_threads=1，预热=10
- 目标产物：两个分步脚本、一个绘图脚本；所有 .keras/.tflite 存 edge_optimized_models/，JSON 存 reports/，图片存 charts/

请基于以上上下文，需要给出：
0) 过渡使用的conda环境相关配置；
1) part3_edge_prune_only.py（tf_keras+tfmot）；
2) part3_edge_quant_arch_nas.py（Keras 3：PTQ、Depthwise-Student、NAS、TFLite 评测）；
3) part3_edge_plot.py（生成稀疏度-精度曲线、量化对比图、延迟-精度帕累托图）；
4) 若 INT8 不支持时的回退逻辑与错误标注。

但在此之前，请先确认是否清晰了这次项目内容，必要时可以联网确认一些包的版本关系。

运行命令：
python part3_convert_eval_legacy.py --arch depthwise \
  --width 0.75 --depth 1.0 \
  --weights edge_optimized_models/depthwise_w0_75_d1_0.weights.h5 \
  --int8