# MLS-A2 — Multi-Scale Optimization

## Envs
- macOS (Apple Silicon), Python 3.10
- TensorFlow / TFLite (CPU only for evaluation)
- Note: Keras3→TFLite need firstly `.keras` → `.h5` then convert.

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
- scripts - Demo Materials

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

### Key Reference
- Part 4 Report (Final Indicators, Pareto, Recommendations): ` multi_stcale_optimize_report. json `.
- Real indicators for each route in the cloud: ` cloud_optimize_results. json `.
- Pruning and fine-tuning records: ` edge_pruning.results. json `.
- 3 Metrics in NAS Quantification Version  (Size/Latency/Accuracy): ` edge_quant_desults_legacy. json `.
- NAS proxy search candidate: ` nas_dearch_desults. json`.  
- Final depthwise training record: ` train_depthwises_taught. json `.
- Project summary of P2/P3 structure (explanation of parallel route): `P23intro.md`.

---
## Example inference scripts

Cloud：python scripts/infer_cloud.py --model cloud_optimized_models/mp_cnn_ep15.keras
- Zsh: ACC=0.7706  LAT(ms)=28.1980
Edge：python scripts/infer_tflite.py --model edge_optimized_models/depthwise_w0_75_d1_0_drq.tflite --n 5000 --runs 100 --warmup 20
- Zsh: ACC=0.6130  LAT(ms)=0.097439
Tiny：python scripts/infer_tflite.py --model edge_optimized_models/depthwise_w0_75_d1_0_int8.tflite --n 5000 --runs 100 --warmup 20
- Zsh: ACC=0.6150  LAT(ms)=0.064756


## Performance benchmarking scripts
Cloud：python scripts/bench_model.py --type keras --path cloud_optimized_models/mp_cnn_ep15.keras --out bench.csv
- Zsh: size(MB)=6.602265  acc=0.7706  latency(ms)=27.299196  mem(MB)=13.216818
Edge：python scripts/bench_model.py --type tflite --path edge_optimized_models/depthwise_w0_75_d1_0_drq.tflite --out bench.csv
- Zsh: size(MB)=0.059056  acc=0.6130  latency(ms)=0.118652  mem(MB)=1.059056
Tiny：python scripts/bench_model.py --type tflite --path edge_optimized_models/depthwise_w0_75_d1_0_int8.tflite --out bench.csv
- Zsh: size(MB)=0.062144  acc=0.6150  latency(ms)=0.070756  mem(MB)=1.062144
