import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

REPORTS = Path("reports")
CHARTS = Path("charts")

def plot_quantization_comparison():
    """
    Plots a comparison of different quantization methods (DRQ, FP16, INT8)
    in terms of model size, latency, and accuracy.
    """
    with open(REPORTS / "edge_quant_results_legacy.json") as f:
        data = json.load(f)["tflite"]

    labels = ['DRQ', 'FP16', 'INT8']
    sizes = [data['drq']['size_mb'], data['fp16']['size_mb'], data['int8']['size_mb']]
    latencies = [data['drq']['latency_ms_per_sample'], data['fp16']['latency_ms_per_sample'], data['int8']['latency_ms_per_sample']]
    accuracies = [data['drq']['accuracy'], data['fp16']['accuracy'], data['int8']['accuracy']]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for size and latency
    rects1 = ax1.bar(x - width/2, sizes, width, label='Size (MB)')
    rects2 = ax1.bar(x + width/2, latencies, width, label='Latency (ms)')

    ax1.set_ylabel('Size (MB) / Latency (ms)')
    ax1.set_title('Quantization Comparison: Size, Latency, and Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(loc='upper left')

    # Line chart for accuracy on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, accuracies, color='red', marker='o', linestyle='--', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig(CHARTS / "part3_quantization_comparison.png")
    plt.close()

def plot_latency_vs_accuracy_pareto():
    """
    Plots the Pareto front of latency vs. accuracy for the NAS results.
    """
    with open(REPORTS / "nas_search_results.json") as f:
        results = json.load(f)

    points = []
    # The results are in the "candidates" field
    for r in results["candidates"]:
        name = f"w{r['width_mult']}_d{r['depth_mult']}"
        # Using params as a proxy for latency for now
        latency = r['params'] 
        accuracy = r['eval_acc']
        points.append((name, latency, accuracy))

    points.sort(key=lambda x: x[1])

    pareto_front = [points[0]]
    for pt in points[1:]:
        if pt[2] >= pareto_front[-1][2]:
            pareto_front.append(pt)
    
    pareto_lats = [p[1] for p in pareto_front]
    pareto_accs = [p[2] for p in pareto_front]

    plt.figure(figsize=(10, 6))
    plt.plot(pareto_lats, pareto_accs, marker='o', linestyle='-', color='r', label='Pareto Front')

    for name, lat, acc in points:
        is_pareto = any(p[1] == lat and p[2] == acc for p in pareto_front)
        plt.scatter(lat, acc, 
                    marker='x' if not is_pareto else 'o', 
                    color='blue' if not is_pareto else 'red', 
                    label=name if is_pareto else None)
        plt.text(lat, acc, f' {name}', fontsize=9)

    plt.title('Params vs. Accuracy for NAS models')
    plt.xlabel('Parameters')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(CHARTS / "part3_latency_vs_accuracy_pareto.png")
    plt.close()


def plot_sparsity_vs_accuracy():
    """
    Plots the relationship between model sparsity and accuracy based on pruning results.
    """
    with open(REPORTS / "edge_pruning_results.json") as f:
        data = json.load(f)

    sparsities = []
    accuracies = []
    for key, value in data.items():
        if key.startswith("student_pruned_s"):
            sparsities.append(float(key.split('_s')[-1]))
            accuracies.append(value["accuracy"])

    plt.figure(figsize=(10, 6))
    plt.plot(sparsities, accuracies, marker='o', linestyle='-')
    plt.title('Sparsity vs. Accuracy')
    plt.xlabel('Sparsity Level')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(CHARTS / "part3_sparsity_vs_accuracy.png")
    plt.close()


if __name__ == "__main__":
    plot_quantization_comparison()
    plot_latency_vs_accuracy_pareto()
    plot_sparsity_vs_accuracy()
