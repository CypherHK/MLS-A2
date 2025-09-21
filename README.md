# MLS-A2 — Multi-Scale Optimization

## Envs
- macOS (Apple Silicon), Python 3.10
- TensorFlow / TFLite (CPU only for evaluation)
- Note: Keras3→TFLite 在 M1 需先 `.keras` → `.h5` 再转换

## Repo Layout
- part1_baseline.py — Baseline CNN (train/eval)
- part2_cloud_mp_only.py — Mixed Precision
- part2_cloud_rest_only.py — Distributed / Batch Accum / KD
- part3_nas_only.py — NAS (proxy)
- part3_train_depthwise_selected.py — Train selected depthwise model
- part3_edge_prune_only.py — Prune + Finetune (student CNN)
- part3_convert_eval_legacy.py — Keras→TFLite + evaluation
- part4_deployment_pipeline.py — Multi-scale deployment pipeline
- reports/ — JSON metrics (cloud, edge, nas, pruning, pipeline)
- cloud_optimized_models/, edge_optimized_models/ — Saved models & TFLites
- charts/ — Figures

## Quick Start
1) Prepare models & reports (already provided under `/reports` and model dirs).
2) Run multi-scale pipeline:
```bash
python part4_deployment_pipeline.py
Outputs: multi_scale_optimization_report.json
```
3) Make Part5 charts:
```bash
python make_part5_charts.py
Outputs: charts/accuracy_vs_size.png, charts/accuracy_vs_latency.png
```
4) Demo Notebook:
Open demo_notebook.ipynb → run all cells.
## Notes
Latency measured on CPU, single thread; absolute values vary by device.
Power values are target budgets (Cloud50W, Edge2W, MCU~10mW), not measurements.


---

### 关键引用（与结论/数值对应）
- P4 报告（最终指标、Pareto、建议）：`multi_scale_optimization_report.json`。
- 云端各路线真实指标：`cloud_optimization_results.json`。
- 剪枝与微调记录：`edge_pruning_results.json`。
- NAS 量化三版本指标（尺寸/延迟/精度）：`edge_quant_results_legacy.json`。
- NAS 代理搜索候选：`nas_search_results.json`。 
- 最终 depthwise 训练记录：`train_depthwise_selected.json`。
- 你项目对 P2/P3 的结构总结（并行路线的说明）：`p23介绍.md`。

---
