# part3_edge_plot.py
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"
CHARTS  = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

def J(p): 
    if p.exists():
        return json.loads(p.read_text())
    return {}

def pruning_curve():
    data = J(REPORTS / "edge_prune_quant.json").get("pruning", {})
    if not data: 
        print("[skip] pruning curve"); return
    xs, ya, ys, yl = [], [], [], []
    for k, v in data.items():
        sp = int(k.split("_")[-1]) / 100.0
        xs.append(sp)
        ya.append(v.get("test_accuracy"))
        ys.append(v.get("tflite_fp32_size_mb"))
        yl.append(v.get("tflite_fp32_latency_ms"))
    # 精度曲线
    fig = plt.figure(figsize=(6,4))
    plt.plot(xs, ya, marker="o")
    plt.xlabel("Sparsity"); plt.ylabel("Keras Test Accuracy")
    plt.title("Pruning: Sparsity vs Accuracy")
    plt.tight_layout(); fig.savefig(CHARTS / "part3_pruning_sparsity_vs_accuracy.png"); plt.close(fig)

def quant_bar():
    data = J(REPORTS / "edge_prune_quant.json").get("quantization", {})
    if not data: 
        print("[skip] quant comparison"); return
    for name, pack in data.items():
        labels = []; acc = []; size = []; lat = []
        for q in ["dynamic","fp16","int8"]:
            if q in pack:
                labels.append(q.upper())
                acc.append(pack[q].get("accuracy"))
                size.append(pack[q].get("size_mb"))
                lat.append(pack[q].get("latency_ms"))
        # 精度
        fig = plt.figure(figsize=(6,4))
        plt.bar(labels, acc); plt.ylim(0,1.0)
        plt.ylabel("TFLite Accuracy"); plt.title(f"Quantization Comparison ({name})")
        plt.tight_layout(); fig.savefig(CHARTS / f"part3_quant_{name}_accuracy.png"); plt.close(fig)
        # 大小
        fig = plt.figure(figsize=(6,4))
        plt.bar(labels, size)
        plt.ylabel("Model Size (MB)"); plt.title(f"Quantization Size ({name})")
        plt.tight_layout(); fig.savefig(CHARTS / f"part3_quant_{name}_size.png"); plt.close(fig)
        # 延迟
        fig = plt.figure(figsize=(6,4))
        plt.bar(labels, lat)
        plt.ylabel("Latency (ms/sample)"); plt.title(f"Quantization Latency ({name})")
        plt.tight_layout(); fig.savefig(CHARTS / f"part3_quant_{name}_latency.png"); plt.close(fig)

def pareto_latency_acc():
    arch = J(REPORTS / "edge_arch_nas.json")
    points = []
    # Depthwise-Student（三种量化里选一个展示，这里挑 INT8 优先，否则 FP16）
    t = arch.get("depthwise_student", {}).get("tflite", {})
    if "int8" in t:
        points.append(("Depthwise-Student INT8", t["int8"]["latency_ms_per_sample"], t["int8"]["accuracy"]))
    elif "fp16" in t:
        points.append(("Depthwise-Student FP16", t["fp16"]["latency_ms_per_sample"], t["fp16"]["accuracy"]))
    # NAS 候选（FP16）
    for r in arch.get("nas", []):
        points.append((f"NAS w{r['width_mult']},d{r['depth_mult']}",
                       r["tflite_fp16_latency_ms"], r["tflite_fp16_accuracy"]))
    if not points:
        print("[skip] latency-accuracy Pareto"); return
    fig = plt.figure(figsize=(6,4))
    xs = [p[1] for p in points]; ys = [p[2] for p in points]
    plt.scatter(xs, ys)
    for (name, x, y) in points:
        plt.annotate(name, (x, y))
    plt.xlabel("Latency (ms/sample)"); plt.ylabel("TFLite Accuracy")
    plt.title("Latency vs Accuracy (Edge Candidates)")
    plt.tight_layout(); fig.savefig(CHARTS / "part3_latency_vs_accuracy.png"); plt.close(fig)

def main():
    pruning_curve()
    quant_bar()
    pareto_latency_acc()
    print("[done] charts in charts/")

if __name__ == "__main__":
    main()
