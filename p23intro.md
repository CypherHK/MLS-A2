# Parts 2–3 Summary: Cloud- and Edge-Side Model Optimization

* **Part 2: Cloud-Side Optimization** — Leverages powerful cloud compute to efficiently train a high-performing baseline model via mixed precision (MP), distributed training, gradient accumulation, and knowledge distillation (KD).
* **Part 3: Edge-Side Optimization** — Makes the model smaller and faster for resource-constrained devices (phones, embedded, IoT). Covers neural architecture search (NAS), pruning, and quantization.

---

# Model Architecture Analysis

The project primarily uses two core architectures: a standard CNN and a depthwise-separable CNN optimized for efficiency.

## 1) Standard CNN Model

* **Role**:
  The base model for all cloud optimizations in Part 2, and the “student” in the Part 3 pruning pipeline.
* **Definition scripts**:
  `part2_cloud_rest_only.py: build_cnn()`
  `part3_edge_prune_only.py: create_student_architecture()`
* **Design (CIFAR-10)**: Intentionally over-parameterized to leave headroom for pruning/quantization.
  **Structure**:

  1. **Input**: 32×32×3 images
  2. **Three conv blocks**, each:

     ```
     Conv2D → BatchNorm → ReLU
     Conv2D → BatchNorm → ReLU
     MaxPooling2D
     ```

     Number of filters per block: 32, 64, 128
  3. **Head**:

     ```
     GlobalAveragePooling2D
     Dropout(0.5)
     Dense(256, activation="relu")
     Dropout(0.3)
     Dense(10)       # logits for 10 classes
     ```

## 2) Depthwise-Separable CNN

* **Role**:
  The dedicated edge-efficient architecture in Part 3, used for NAS, full training, and quantization (MobileNet-style).
* **Definition scripts**:
  `part3_nas_only.py: build_depthwise_student()`
  `part3_train_depthwise_selected.py: build_depthwise_student()`
* **Design**:

  1. **Input**: 32×32×3
  2. **Three depthwise-separable blocks**, each:

     ```
     DepthwiseConv2D(3×3) → BatchNorm → ReLU
     Conv2D(1×1, pointwise) → BatchNorm → ReLU
     MaxPooling2D
     ```

     This factorization drastically reduces parameters and FLOPs.
  3. **Searchable hyperparameters**: width\_mult, depth\_mult (the NAS targets).
  4. **Head**: Similar to the standard CNN but with adjusted units/dropout.

### Architecture Summary

* **Are Part 2 & Part 3 architectures identical?** Not exactly.

  * The **standard CNN** from Part 2 is also used by **pruning** in Part 3 (as the KD student).
  * **NAS + full training + quantization** in Part 3 uses a **separate, more efficient depthwise model**.
* **Logic**: Train a strong **student CNN via KD**, then split into two tracks:
  (a) **Prune** the student CNN, and
  (b) **Design/train** an inherently edge-friendly **depthwise** model.

---

## Part 2: Cloud-Side Optimization

**Goal**: Speed up training and produce a strong “student” baseline for subsequent optimization.

### 1) Mixed Precision (MP)

* **What**: Use `float16` + `float32` to speed up training and reduce memory on supported hardware.
* **Core script**: `part2_cloud_mp_only.py`
* **How**: In `main`, set `mixed_precision.set_global_policy("mixed_float16")`.
* **Base model**: `build_cnn()`
* **Artifacts**:

  * Model: `cloud_optimized_models/mp_cnn_ep15.keras`
  * Reports: `reports/cloud_mp.json`, `reports/history_mp.json`
* **Result**: Records training time and final accuracy under MP, confirming acceleration.

### 2) Distributed Training

* **What**: Simulated data parallelism to scale training and reduce wall time.
* **Core script**: `part2_cloud_rest_only.py`
* **How**: In `main`, use `tf.distribute.MirroredStrategy`.
* **Base model**: `build_cnn()`
* **Artifacts**:

  * Model: `cloud_optimized_models/distributed_fp32_ep5.keras`
  * Reports: `reports/cloud_rest.json` (field `distributed`), `reports/history_distributed.json`
* **Result**: Total and per-epoch timings plus evaluation metrics.

### 3) Gradient Accumulation

* **What**: Simulates large batch training under limited memory by accumulating gradients across micro-batches.
* **Core script**: `part2_cloud_rest_only.py`
* **How**: Custom `AccumModel` overriding `train_step`.
* **Base model**: `build_cnn()`
* **Artifacts**:

  * Model: `cloud_optimized_models/batch_accum_fp32_ep10.keras`
  * Reports: `reports/cloud_rest.json` (field `batch_processing`), `reports/history_batch_accum.json`
* **Result**: Timing and performance under accumulation; proves usefulness in constrained settings.

### 4) Knowledge Distillation (KD)

* **What**: Train a larger **teacher** and distill to a lighter **student** via soft labels.
* **Core script**: `part2_cloud_rest_only.py`
* **How**: Custom `Distiller` combining student loss and distillation loss in `train_step`.
* **Base models**:

  * **Teacher**: `build_cnn(width_mult=1.6, …)`
  * **Student**: `build_cnn()`
* **Artifacts**:

  * Models:
    `cloud_optimized_models/kd_teacher_ep16.keras` (teacher)
    `cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.keras` (student)
    and `.weights.h5`
  * Reports: `reports/cloud_rest.json` (field `knowledge_distillation`),
    `reports/history_kd_teacher.json`, `reports/history_kd_student.json`
* **Result**: A compact, high-performing **student** that becomes the **starting point for pruning in Part 3**.

---

## Part 3: Edge-Side Optimization

**Goal**: Starting from the KD student (for pruning) or from a freshly designed efficient model (for NAS), further optimize into ultra-light models suitable for edge deployment.
*(Only the **pruning** track uses the Part 2 KD student; NAS + full training + quantization uses the **NAS-selected depthwise** model.)*

### 1) Neural Architecture Search (NAS)

* **What**: Explore width/depth in a defined search space using proxy training to quickly estimate candidate performance.
* **Core script**: `part3_nas_only.py`
* **How**: In `main`, loop over configs, call `build_depthwise_student` to train/evaluate candidates.
* **Base model**: Dynamically built by `build_depthwise_student`.
* **Artifacts**:

  * Report: `reports/nas_search_results.json`
* **Result**: A list of candidates with proxy accuracy and parameter counts to inform selection.

### 2) Train the Selected Depthwise Model

* **What**: Pick optimal width/depth from NAS or expert prior; run full training and save weights.
* **Core script**: `part3_train_depthwise_selected.py`
* **How**: `main` receives hyper-params, calls `build_depthwise_student`, then trains fully.
* **Base model**: The selected depthwise configuration.
* **Artifacts**:

  * Models: `edge_optimized_models/depthwise_w{...}_d{...}.keras` and `.weights.h5`
  * Report: `reports/train_depthwise_selected.json`
* **Result**: A well-trained depthwise model ready for quantization (or pruning, if desired).

### 3) Pruning

* **What**: Magnitude pruning with fine-tuning to reduce size/compute; implemented via TF-MOT.
* **Core script**: `part3_edge_prune_only.py`
* **How**: `prune_and_finetune` wraps the model with `tfmot.sparsity.keras.prune_low_magnitude` and retrains.
* **Base model**: KD student from Part 2 (`cloud_optimized_models/kd_student_ep16_alpha0_7_T2_0.weights.h5`)
* **Artifacts**:

  * Models: `edge_optimized_models/student_pruned_s{...}.keras` (multiple sparsity levels)
  * Report: `reports/edge_pruning_results.json`
* **Result**: A family of sparse models with target/actual sparsity and accuracy, useful for further compression or deployment.

### 4) Quantization & Evaluation

* **What**: Convert trained Keras models to TFLite and apply DRQ, FP16, and INT8 quantization; measure **size, latency, accuracy** on CPU precisely.
* **Core script**: `part3_convert_eval_legacy.py`
* **How**:

  * `convert_all`: produce `{drq, fp16, int8}` from Keras
  * `eval_tflite`: CPU inference to measure single-sample latency & accuracy
* **Base model**: Passed via `--weights` (`.h5`), loaded into student/depthwise models.
* **Artifacts**:

  * Models:
    `edge_optimized_models/..._drq.tflite`, `..._fp16.tflite`, `..._int8.tflite`
  * Report: `reports/edge_quant_results_legacy.json`
* **Result**: Deployable TFLite models and a final comparison report showing the **trade-offs among DRQ/FP16/INT8** in size, latency, and accuracy.

**Example command**:

```bash
python part3_convert_eval_legacy.py --arch depthwise \
  --width 0.75 --depth 1.0 \
  --weights edge_optimized_models/depthwise_w0_75_d1_0.weights.h5 \
  --int8
```
