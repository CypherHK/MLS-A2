# make_part5_charts.py
import os, json
import pandas as pd
import matplotlib.pyplot as plt

REPORT = "multi_scale_optimization_report.json"
OUTDIR = "charts"
os.makedirs(OUTDIR, exist_ok=True)

with open(REPORT, "r") as f:
    rep = json.load(f)

rows = []
for target, d in rep["optimization_results"].items():
    rows.append({
        "Target": target,
        "Strategy": d["optimization_strategy"],
        "Model Path": d["model_path"],
        "Model Size (MB)": float(d["model_size_mb"]),
        "Accuracy": float(d["accuracy"]),
        "Latency (ms)": float(d["estimated_latency_ms"]),
        "Memory (MB)": float(d["memory_usage_mb"]),
    })
df = pd.DataFrame(rows)

# Accuracy vs. Size
plt.figure()
plt.scatter(df["Model Size (MB)"], df["Accuracy"])
for _, r in df.iterrows():
    plt.text(r["Model Size (MB)"], r["Accuracy"], r["Target"])
plt.xlabel("Model Size (MB)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Model Size")
plt.savefig(os.path.join(OUTDIR, "accuracy_vs_size.png"), bbox_inches="tight")
plt.close()

# Accuracy vs. Latency
plt.figure()
plt.scatter(df["Latency (ms)"], df["Accuracy"])
for _, r in df.iterrows():
    plt.text(r["Latency (ms)"], r["Accuracy"], r["Target"])
plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Latency")
plt.savefig(os.path.join(OUTDIR, "accuracy_vs_latency.png"), bbox_inches="tight")
plt.close()

print("Saved to:", os.path.join(OUTDIR, "accuracy_vs_size.png"))
print("Saved to:", os.path.join(OUTDIR, "accuracy_vs_latency.png"))
