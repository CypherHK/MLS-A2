# plot_part2.py
# 读取 Part 2 的 JSON 结果（cloud_optimization_results.json 或 cloud_mp.json + cloud_rest.json）
# 生成：总览精度图、总览耗时图、KD 参数量-精度散点图
# 可选：若存在以下 history JSON，将额外生成 4 张训练曲线图：
#   reports/history_distributed.json        -> charts/part2_distributed_curve.png
#   reports/history_batch_accum.json        -> charts/part2_batch_accum_curve.png
#   reports/history_kd_teacher.json         -> charts/part2_kd_teacher_curve.png
#   reports/history_kd_student.json         -> charts/part2_kd_student_curve.png

import json, sys
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"
CHARTS = ROOT / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

def _read_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def load_results():
    merged = _read_json(REPORTS / "cloud_optimization_results.json")
    if merged:
        return merged, "cloud_optimization_results.json"
    mp = _read_json(REPORTS / "cloud_mp.json")
    rest = _read_json(REPORTS / "cloud_rest.json")
    if mp and rest:
        return {**mp, **rest}, "cloud_mp.json + cloud_rest.json"
    return None, None

def plot_overview(results):
    # --- Accuracy overview ---
    names, accs = [], []
    if "mixed_precision" in results:
        names.append("MixedPrecision"); accs.append(results["mixed_precision"].get("test_accuracy"))
    if "distributed" in results:
        names.append("Distributed"); accs.append(results["distributed"].get("test_accuracy"))
    if "batch_processing" in results:
        names.append("BatchAccum"); accs.append(results["batch_processing"].get("test_accuracy"))
    if "knowledge_distillation" in results:
        kd = results["knowledge_distillation"]
        if "teacher_accuracy" in kd:
            names.append("KD-Teacher"); accs.append(kd.get("teacher_accuracy"))
        if "student_accuracy" in kd:
            names.append("KD-Student"); accs.append(kd.get("student_accuracy"))
    if names:
        fig = plt.figure(figsize=(6,4))
        plt.bar(names, accs)
        plt.ylabel("Test Accuracy")
        plt.title("Part 2: Accuracy Overview")
        plt.xticks(rotation=15)
        plt.tight_layout()
        fig.savefig(CHARTS / "part2_overview_accuracy.png")
        plt.close(fig)

    # --- Time per epoch overview ---
    names_t, times = [], []
    if "mixed_precision" in results:
        names_t.append("MixedPrecision"); times.append(results["mixed_precision"].get("time_per_epoch_s"))
    if "distributed" in results:
        names_t.append("Distributed"); times.append(results["distributed"].get("time_per_epoch_s"))
    if "batch_processing" in results:
        names_t.append("BatchAccum"); times.append(results["batch_processing"].get("time_per_epoch_s"))
    if names_t:
        fig = plt.figure(figsize=(6,4))
        plt.bar(names_t, times)
        plt.ylabel("Time per epoch (s)")
        plt.title("Part 2: Time per Epoch")
        plt.xticks(rotation=15)
        plt.tight_layout()
        fig.savefig(CHARTS / "part2_overview_time.png")
        plt.close(fig)

def plot_kd_params_vs_acc(results):
    kd = results.get("knowledge_distillation")
    if not kd or "teacher_params" not in kd or "student_params" not in kd:
        return
    x = [kd["teacher_params"], kd["student_params"]]
    y = [kd.get("teacher_accuracy"), kd.get("student_accuracy")]
    labels = ["Teacher", "Student"]
    fig = plt.figure(figsize=(6,4))
    plt.scatter(x, y)
    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi))
    plt.xlabel("Parameters (#)")
    plt.ylabel("Test Accuracy")
    plt.title("KD: Params vs Accuracy")
    plt.tight_layout()
    fig.savefig(CHARTS / "part2_kd_params_vs_acc.png")
    plt.close(fig)

def _try_plot_curve(history_path: Path, out_png: Path, title: str):
    hist = _read_json(history_path)
    if not hist:
        print(f"[skip] no history: {history_path}")
        return False
    # 兼容不同键名
    train_acc = hist.get("accuracy") or hist.get("sparse_categorical_accuracy")
    val_acc = hist.get("val_accuracy") or hist.get("val_sparse_categorical_accuracy")
    if (train_acc is None) and (val_acc is None):
        print(f"[skip] no accuracy-like keys in {history_path.name}")
        return False
    fig = plt.figure(figsize=(6,4))
    if train_acc is not None: plt.plot(train_acc, label="train_acc")
    if val_acc is not None: plt.plot(val_acc, label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(title); plt.legend()
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True

def plot_curves_if_available():
    pairs = [
        (REPORTS / "history_distributed.json",     CHARTS / "part2_distributed_curve.png",   "Distributed Training Curve"),
        (REPORTS / "history_batch_accum.json",     CHARTS / "part2_batch_accum_curve.png",   "Batch Accumulation Training Curve"),
        (REPORTS / "history_kd_teacher.json",      CHARTS / "part2_kd_teacher_curve.png",    "KD Teacher Training Curve"),
        (REPORTS / "history_kd_student.json",      CHARTS / "part2_kd_student_curve.png",    "KD Student Training Curve"),
    ]
    for src, dst, title in pairs:
        _try_plot_curve(src, dst, title)

def main():
    results, source = load_results()
    if not results:
        print("No results found. Put results in reports/cloud_optimization_results.json, or cloud_mp.json + cloud_rest.json.")
        sys.exit(1)
    print(f"[info] Loaded results from: {source}")
    plot_overview(results)
    plot_kd_params_vs_acc(results)
    plot_curves_if_available()
    print(f"[done] Charts written to: {CHARTS}")

if __name__ == "__main__":
    main()
