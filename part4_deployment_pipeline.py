# part4_deployment_pipeline.py
import os, json, time, glob
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import tensorflow as tf

# ----------- 环境/设备设定：在 Mac M1 上只用 CPU，避免金属后端带来的波动 -----------
def _pin_cpu_only():
    try:
        tf.config.set_visible_devices([], "GPU")
        tf.config.set_visible_devices([], "TPU")
    except Exception:
        pass

_pin_cpu_only()

# -------------------- 通用数据结构（与作业骨架一致） --------------------
@dataclass
class DeploymentTarget:
    name: str
    max_model_size_mb: float
    max_latency_ms: float
    max_memory_mb: float
    power_budget_mw: float
    compute_capability: str  # 'cloud', 'edge', 'tiny'

@dataclass
class OptimizationResult:
    model_path: str
    accuracy: float
    model_size_mb: float
    estimated_latency_ms: float
    memory_usage_mb: float
    optimization_strategy: str

# -------------------- 工具函数 --------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def get_model_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6

def load_cifar10_test(num_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    # 与你现有流程一致：32x32x3、[0,1] 归一化
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.reshape(-1)
    if num_samples is not None:
        x_test, y_test = x_test[:num_samples], y_test[:num_samples]
    return x_test, y_test

def evaluate_keras_latency_accuracy(model: tf.keras.Model, x: np.ndarray, y: np.ndarray,
                                    warmup: int = 10, runs: int = 100, batch_size: int = 1) -> Tuple[float, float]:
    # 精度
    y_prob = model.predict(x, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc = float(np.mean(y_pred == y))
    # 单样本延迟
    sample = x[:batch_size]
    for _ in range(warmup):
        _ = model.predict(sample, batch_size=batch_size, verbose=0)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model.predict(sample, batch_size=batch_size, verbose=0)
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) / runs * 1000.0
    return acc, float(lat_ms)

def _quantize_input_like(interpreter, x_batch: np.ndarray) -> np.ndarray:
    d = interpreter.get_input_details()[0]
    scale, zero = d["quantization"]
    dtype = d["dtype"]
    if scale == 0:  # 未量化输入
        return x_batch.astype(dtype)
    q = np.round(x_batch / scale + zero).astype(dtype)
    # 按 dtype 范围裁剪
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        q = np.clip(q, info.min, info.max)
    return q

def evaluate_tflite_latency_accuracy(tflite_path: str, x: np.ndarray, y: np.ndarray,
                                     warmup: int = 20, runs: int = 100, batch_size: int = 1) -> Tuple[float, float, float]:
    # 统一只用 CPU，线程=1，保证 Mac M1 可复现
    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 估算内存（粗略）：模型 + I/O + 1MB 运行开销
    model_mb = get_model_size_mb(tflite_path)
    io_bytes = 0
    for d in input_details + output_details:
        nbytes = np.prod(d["shape"]) * np.dtype(d["dtype"]).itemsize
        io_bytes += int(nbytes)
    est_ram_mb = model_mb + io_bytes / 1e6 + 1.0

    # 延迟（单样本）
    xb = x[:batch_size]
    inp = _quantize_input_like(interpreter, xb)
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
    t0 = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) / runs * 1000.0

    # 精度（默认评 2000 样本以加速）
    N = min(2000, x.shape[0])
    correct = 0
    for i in range(N):
        xi = _quantize_input_like(interpreter, x[i:i+1])
        interpreter.set_tensor(input_details[0]["index"], xi)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]["index"])
        if int(np.argmax(logits, axis=1)[0]) == int(y[i]):
            correct += 1
    acc = correct / N
    return float(acc), float(lat_ms), float(est_ram_mb)

# -------------------- Optimizers（以“读取现有产物+评测/筛选”为主） --------------------
class ModelOptimizer:
    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        raise NotImplementedError

class CloudOptimizer(ModelOptimizer):
    """选择 cloud_optimized_models 中精度最高的 .keras；若无则用 baseline。"""
    def __init__(self, cloud_dir="cloud_optimized_models", reports_dir="reports"):
        self.cloud_dir = cloud_dir
        self.reports_dir = reports_dir

    def _collect_from_reports(self) -> List[Tuple[str, float]]:
        cands = []
        p = os.path.join(self.reports_dir, "cloud_optimization_results.json")
        if os.path.exists(p):
            j = json.load(open(p, "r"))

        # 从技术路线级 JSON 映射到你已有的 .keras 文件名
        mapping = {
            "mp_cnn_ep15.keras": j.get("mixed_precision", {}).get("test_accuracy"),
            "distributed_fp32_ep5.keras": j.get("distributed", {}).get("test_accuracy"),
            "batch_accum_fp32_ep10.keras": j.get("batch_processing", {}).get("test_accuracy"),
            "kd_student_ep16_alpha0_7_T2_0.keras": j.get("knowledge_distillation", {}).get("student_accuracy"),
            "kd_teacher_ep16.keras": j.get("knowledge_distillation", {}).get("teacher_accuracy"),
        }  # 来自 cloud_optimization_results.json 的结构。:contentReference[oaicite:5]{index=5}

        for fname, acc in mapping.items():
            if acc is None:
                continue
            fpath = os.path.join(self.cloud_dir, fname)
            if os.path.exists(fpath):
                cands.append((fpath, float(acc)))

    # 若没有匹配到，就回退到目录扫描
        if not cands:
            for f in glob.glob(os.path.join(self.cloud_dir, "*.keras")):
                cands.append((f, -1.0))  # 没有精度信息时标 -1

        return cands


    def _pick_best(self) -> Optional[str]:
        cands = self._collect_from_reports()
        if not cands:
            for f in glob.glob(os.path.join(self.cloud_dir, "*.keras")):
                cands.append((f, -1.0))
        if not cands:
            return None
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[0][0]

    def optimize(self, base_model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        x_test, y_test = load_cifar10_test(5000)
        chosen = self._pick_best()
        if chosen and os.path.exists(chosen):
            try:
                model = tf.keras.models.load_model(chosen)
                strategy = f"cloud:{os.path.basename(chosen)}"
                save_path = chosen
            except Exception:
                model = base_model
                strategy = "cloud:baseline_fallback"
                save_path = "baseline_model.keras"
        else:
            model = base_model
            strategy = "cloud:baseline"
            save_path = "baseline_model.keras"

        acc, lat = evaluate_keras_latency_accuracy(model, x_test, y_test, batch_size=1)
        size_mb = get_model_size_mb(save_path)
        # 内存估算：2×模型大小 + 输入张量开销
        mem_mb = size_mb * 2 + (32*32*3*4)/1e6
        return OptimizationResult(
            model_path=save_path,
            accuracy=acc,
            model_size_mb=size_mb,
            estimated_latency_ms=lat,
            memory_usage_mb=mem_mb,
            optimization_strategy=strategy
        )

class EdgeOptimizer(ModelOptimizer):
    """优先使用 reports/edge_quant_results_legacy.json；否则扫描 edge_optimized_models/*.tflite 并实测。"""
    def __init__(self, edge_dir="edge_optimized_models", reports_dir="reports"):
        self.edge_dir = edge_dir
        self.reports_dir = reports_dir

    def _read_quant_report(self) -> List[Dict[str, Any]]:
        p = os.path.join(self.reports_dir, "edge_quant_results_legacy.json")
        if not os.path.exists(p):
            return []
        try:
            j = json.load(open(p, "r"))
            items = []

        # 先从 weights_path 提取公共前缀（更鲁棒，不受浮点字符串影响）
        # 例如：.../edge_optimized_models/depthwise_w0_75_d1_0.weights.h5 → depthwise_w0_75_d1_0
            weights_path = j.get("weights_path", "")
            if weights_path:
                stem = os.path.basename(weights_path).split(".weights")[0]
            else:
            # 退化方案：由 width/depth 拼接（与目录一致）
                w = j.get("width")
                d = j.get("depth")
                stem = f"depthwise_w{str(w).replace('.', '_')}_d{str(d).replace('.', '_')}"

            base = os.path.join(self.edge_dir, stem)  # edge_optimized_models/depthwise_w0_75_d1_0

        # 三种量化；注意 JSON 的延迟键名是 latency_ms_per_sample
            tfl = j.get("tflite", {})
            for q in ["drq", "fp16", "int8"]:
                qj = tfl.get(q)
                if not qj:
                    continue
                items.append({
                    "model_path": f"{base}_{q}.tflite",
                    "accuracy": float(qj.get("accuracy")) if qj.get("accuracy") is not None else None,
                    "latency_ms": float(qj.get("latency_ms_per_sample")) if qj.get("latency_ms_per_sample") is not None else None, 
                "size_mb": float(qj.get("size_mb")) if qj.get("size_mb") is not None else None
            })
            return items
        except Exception:
            return []

            

    def _scan_tflite(self) -> List[str]:
        # 形如 depthwise_..._{drq|fp16|int8}.tflite
        paths = sorted(glob.glob(os.path.join(self.edge_dir, "*.tflite")))
        return paths

    def _evaluate_paths(self, paths: List[str], x: np.ndarray, y: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        for p in paths:
            acc, lat, mem = evaluate_tflite_latency_accuracy(p, x, y)
            results.append({
                "model_path": p,
                "accuracy": acc,
                "latency_ms": lat,
                "size_mb": get_model_size_mb(p),
                "memory_mb": mem
            })
        return results

    def optimize(self, base_model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        x_test, y_test = load_cifar10_test(6000)

        items = self._read_quant_report()
        # 如果报告里没有延迟/精度，就现场评测一次补齐
        need_eval_paths = []
        for it in items:
            if it.get("latency_ms") is None or it.get("accuracy") is None or it.get("size_mb") is None:
                need_eval_paths.append(it["model_path"])
        if not items:
            need_eval_paths = self._scan_tflite()
            items = [{"model_path": p} for p in need_eval_paths]

        if need_eval_paths:
            evaled = self._evaluate_paths(need_eval_paths, x_test, y_test)
            m = {e["model_path"]: e for e in evaled}
            for it in items:
                if it["model_path"] in m:
                    it.update(m[it["model_path"]])

        # 约束筛选：满足约束内，精度优先；否则选“超约束最少”的
        feasible = [it for it in items
                    if it.get("size_mb", 1e9) <= target.max_model_size_mb
                    and it.get("latency_ms", 1e9) <= target.max_latency_ms]
        if feasible:
            best = sorted(
                feasible,
                key=lambda it: (-it.get("accuracy", -1.0), it.get("latency_ms", 1e9), it.get("size_mb", 1e9))
            )[0]
        else:
            def penalty(it):
                return max(0.0, it.get("size_mb", 1e9) - target.max_model_size_mb) \
                     + 0.01 * max(0.0, it.get("latency_ms", 1e9) - target.max_latency_ms)
            best = sorted(items, key=lambda z: (penalty(z), -z.get("accuracy", -1.0)))[0]

        mem_mb = float(best.get("memory_mb") or (best.get("size_mb", 0.0) + 1.0))
        strat = "tflite:" + (os.path.basename(best["model_path"]).split("_")[-1].split(".")[0])
        return OptimizationResult(
            model_path=best["model_path"],
            accuracy=float(best.get("accuracy", -1.0)),
            model_size_mb=float(best.get("size_mb", get_model_size_mb(best["model_path"]))),
            estimated_latency_ms=float(best.get("latency_ms", -1.0)),
            memory_usage_mb=mem_mb,
            optimization_strategy=strat
        )

class TinyMLOptimizer(ModelOptimizer):
    """Tiny 偏好 INT8 全整型：优先选现成 *_int8*.tflite（尺寸最小者），再评测。"""
    def __init__(self, edge_dir="edge_optimized_models"):
        self.edge_dir = edge_dir

    def _pick_tiny_int8(self) -> Optional[str]:
        paths = glob.glob(os.path.join(self.edge_dir, "*int8*.tflite"))
        if not paths:
            return None
        paths.sort(key=lambda p: get_model_size_mb(p))
        return paths[0]

    def optimize(self, base_model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        x_test, y_test = load_cifar10_test(6000)
        tfl = self._pick_tiny_int8()
        if tfl is None:
            # 极端 fallback：若没有现成 INT8，则退而求其次选任意最小的 tflite
            all_tfl = sorted(glob.glob(os.path.join(self.edge_dir, "*.tflite")), key=lambda p: get_model_size_mb(p))
            if not all_tfl:
                raise FileNotFoundError("No TFLite models found for Tiny target.")
            tfl = all_tfl[0]
        acc, lat, mem = evaluate_tflite_latency_accuracy(tfl, x_test, y_test)
        size_mb = get_model_size_mb(tfl)
        return OptimizationResult(
            model_path=tfl,
            accuracy=acc,
            model_size_mb=size_mb,
            estimated_latency_ms=lat,
            memory_usage_mb=mem,
            optimization_strategy="tflite:int8_prebuilt" if "int8" in os.path.basename(tfl) else "tflite:tiny_fallback"
        )

# -------------------- 多尺度流水线 --------------------
class MultiScaleDeploymentPipeline:
    def __init__(self):
        self.optimizers = {
            "cloud": CloudOptimizer(),
            "edge": EdgeOptimizer(),
            "tiny": TinyMLOptimizer(),
        }
        self.targets = {
            "cloud_server": DeploymentTarget("cloud_server", 1000.0, 100.0, 8000.0, 50000.0, "cloud"),
            "edge_device": DeploymentTarget("edge_device", 50.0, 200.0, 512.0, 2000.0, "edge"),
            "microcontroller": DeploymentTarget("microcontroller", 1.0, 1000.0, 64.0, 10.0, "tiny"),
        }

    def optimize_for_all_targets(self, baseline_model_path: str = "baseline_model.keras") -> Dict[str, OptimizationResult]:
        base_model = tf.keras.models.load_model(baseline_model_path)
        results: Dict[str, OptimizationResult] = {}
        for name, cfg in self.targets.items():
            optimizer = self.optimizers[cfg.compute_capability]
            results[name] = optimizer.optimize(base_model, cfg)
        return results

    @staticmethod
    def _pareto_front(points: List[Dict[str, float]], xkey: str, ykey: str,
                      minimize_x=True, maximize_y=True) -> List[Dict[str, float]]:
        def dominates(a, b):
            bx = (a[xkey] <= b[xkey]) if minimize_x else (a[xkey] >= b[xkey])
            by = (a[ykey] >= b[ykey]) if maximize_y else (a[ykey] <= b[ykey])
            strict = (a[xkey] < b[xkey]) if minimize_x else (a[xkey] > b[xkey])
            strict |= (a[ykey] > b[ykey]) if maximize_y else (a[ykey] < b[ykey])
            return bx and by and strict
        frontier = []
        for i, p in enumerate(points):
            if not any(dominates(q, p) for j, q in enumerate(points) if j != i):
                frontier.append(p)
        frontier.sort(key=lambda z: z[xkey], reverse=not minimize_x)
        return frontier

    def _edge_candidates_from_report(self) -> List[Dict[str, Any]]:
        p = os.path.join("reports", "edge_quant_results_legacy.json")
        out = []
        if not os.path.exists(p):
            return out
        j = json.load(open(p, "r"))

        weights_path = j.get("weights_path", "")
        if weights_path:
            stem = os.path.basename(weights_path).split(".weights")[0]
        else:
            w = j.get("width"); d = j.get("depth")
            stem = f"depthwise_w{str(w).replace('.', '_')}_d{str(d).replace('.', '_')}"

        base = os.path.join("edge_optimized_models", stem)

        for q in ["drq", "fp16", "int8"]:
            qj = j.get("tflite", {}).get(q)
            if not qj:
                continue
            out.append({
            "target": "edge_device",
            "strategy": f"tflite:{q}",
            "model_path": f"{base}_{q}.tflite",
            "size_mb": float(qj.get("size_mb")) if qj.get("size_mb") is not None else None,
            "latency_ms": float(qj.get("latency_ms_per_sample")) if qj.get("latency_ms_per_sample") is not None else None,
            "accuracy": float(qj.get("accuracy")) if qj.get("accuracy") is not None else None
            })
        return out


    def analyze_scaling_trade_offs(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        # 先收集每个 target 的最终选型点
        pts = []
        for k, r in results.items():
            pts.append({
                "target": k,
                "strategy": r.optimization_strategy,
                "model_path": r.model_path,
                "size_mb": r.model_size_mb,
                "latency_ms": r.estimated_latency_ms,
                "accuracy": r.accuracy
            })
        # 并入 Edge 的备选（DRQ/FP16/INT8），提升 Pareto 的信息量
        pts += [it for it in self._edge_candidates_from_report()
                if it["model_path"] not in {p["model_path"] for p in pts}]

        pareto_acc_size = self._pareto_front([p for p in pts if p["accuracy"] is not None],
                                             xkey="size_mb", ykey="accuracy")
        pareto_acc_lat  = self._pareto_front([p for p in pts if p["accuracy"] is not None and p["latency_ms"] is not None],
                                             xkey="latency_ms", ykey="accuracy")

        # 简单瓶颈：看超出约束的主因
        bottlenecks = {}
        for name, tgt in self.targets.items():
            r = results[name]
            bottlenecks[name] = {
                "size_over_mb": max(0.0, r.model_size_mb - tgt.max_model_size_mb),
                "latency_over_ms": max(0.0, r.estimated_latency_ms - tgt.max_latency_ms),
                "memory_over_mb": max(0.0, r.memory_usage_mb - tgt.max_memory_mb),
            }
        return {
            "per_point": pts,
            "pareto_front_accuracy_vs_size": pareto_acc_size,
            "pareto_front_accuracy_vs_latency": pareto_acc_lat,
            "bottlenecks": bottlenecks
        }

    def generate_deployment_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        recs = []
        # A：实时视频（<50ms）
        edge = [p for p in analysis["per_point"] if p["target"] == "edge_device" and p["latency_ms"] is not None]
        if edge and min(e["latency_ms"] for e in edge) <= 50:
            recs.append("A 实时视频：采用 Edge 实时推理（≤50ms），Cloud 作为回传备份/异步再训练。")
        else:
            recs.append("A 实时视频：Edge 延迟仍>50ms，建议更小宽度/深度 + INT8 严格量化；必要时蒸馏提升精度，并保留 Cloud 兜底。")

        # B：IoT（<1mW）→ TinyML
        tiny = [p for p in analysis["per_point"] if p["target"] == "microcontroller"]
        if tiny and tiny[0]["size_mb"] <= 1.0:
            recs.append("B IoT：采用 Tiny INT8 全整型模型（≤1MB），周期性云同步；若精度不足，增加代表性数据进行再校准。")
        else:
            recs.append("B IoT：模型仍偏大/慢，建议更激进剪枝与严格 INT8（仅内建算子），必要时降低输入分辨率。")

        # C：Mobile（Offline）→ Multi-tier
        recs.append("C 移动离线：Multi-tier（本地 FP16/INT8 + 断网缓存 + 联网批量上云），从 Pareto 前沿挑本地最优点。")

        # 通用
        recs.append("通用：沿 Pareto 前沿滚动优化，优先解决各 target 的首要瓶颈（size/latency/memory 超约束最大者）。")
        return recs

# -------------------- 入口 --------------------
def run_multi_scale_optimization(baseline_model_path: str = "baseline_model.keras") -> Dict[str, Any]:
    pipe = MultiScaleDeploymentPipeline()
    results = pipe.optimize_for_all_targets(baseline_model_path)
    analysis = pipe.analyze_scaling_trade_offs(results)
    recs = pipe.generate_deployment_recommendations(analysis)

    report = {
        "optimization_results": {k: asdict(v) for k, v in results.items()},
        "scaling_analysis": analysis,
        "deployment_recommendations": recs
    }
    with open("multi_scale_optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

if __name__ == "__main__":
    rep = run_multi_scale_optimization("baseline_model.keras")
    print("Multi-Scale Optimization Complete!")
    print("Report saved to: multi_scale_optimization_report.json")
