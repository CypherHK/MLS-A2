# Multi-Scale Analysis (Draft)

This document summarizes trade-offs, effectiveness of optimization strategies, and deployment recommendations based on the generated report.

## 1. Multi-Scale Trade-off Table

| Target          | Strategy                | Model Path                                             |   Model Size (MB) |   Accuracy |   Latency (ms) |   Memory Usage (MB) | Power Estimate   | Development Complexity   |
|:----------------|:------------------------|:-------------------------------------------------------|------------------:|-----------:|---------------:|--------------------:|:-----------------|:-------------------------|
| cloud_server    | cloud:mp_cnn_ep15.keras | cloud_optimized_models/mp_cnn_ep15.keras               |            6.6023 |     0.7706 |        27.8076 |             13.2168 | 50.0 W           | Medium                   |
| edge_device     | tflite:drq              | edge_optimized_models/depthwise_w0_75_d1_0_drq.tflite  |            0.0591 |     0.625  |         0.1082 |              1.0591 | 2000 mW          | Low                      |
| microcontroller | tflite:int8_prebuilt    | edge_optimized_models/depthwise_w0_75_d1_0_int8.tflite |            0.0621 |     0.615  |         0.0664 |              1.0652 | 10 mW            | Medium                   |

*Power values are target budgets (not measured).*

## 2. Optimization Strategy Effectiveness

- **Cloud (Mixed Precision, Distributed, Batch Accum, KD):** Mixed precision delivered the best accuracy among cloud variants; distributed and batch accumulation mainly improved throughput during training. KD student trails the teacher but provides a compact baseline for edge pruning.

- **Edge (NAS + Quantization):** DRQ/FP16 retain higher accuracy; INT8 minimizes size/latency for Tiny constraints.

## 3. Deployment Strategy Recommendations

- **A Real-time Video (<50ms):** Edge primary, Cloud backup.
- **B IoT (<1mW):** Tiny INT8 with occasional cloud sync.
- **C Mobile:** Multi-tier (local FP16/INT8 + sync).

## 4. Notes
- All results are evaluated on CPU for stability; absolute latency will vary by device.
