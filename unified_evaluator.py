"""
unified_evaluator.py - Multi-detector, multi-dataset evaluation framework
Compares multiple detection approaches across all available datasets.
"""

import time
import pandas as pd
import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import DatasetLoader
from agents.threat_agent import ThreatAgent
from ml_classifier import MLPromptClassifier
from transformer_classifier import TransformerClassifier
from context_detector import ContextDetector, evaluate_context_vs_baseline
from multi_turn_dataset import generate_multi_turn_dataset

# Suppress verbose agent logging during evaluation (after imports)
logging.getLogger("threat_agent").setLevel(logging.ERROR)
logging.getLogger("ml_classifier").setLevel(logging.WARNING)
logging.getLogger("transformer_classifier").setLevel(logging.WARNING)
logging.getLogger("dataset_loader").setLevel(logging.WARNING)


def _calculate_metrics(y_true, y_pred, latency_ms=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "specificity": round(specificity, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "avg_latency_ms": round(np.mean(latency_ms), 2) if latency_ms else None,
    }


def _batch_evaluate_detector(batch_func, data, name="dataset", sample_size=None, batch_size=64):
    """Evaluate a detector that supports batch processing."""
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    texts = data["text"].tolist()
    y_true = data["label"].tolist()
    y_pred = []
    scores = []
    latencies = []

    start_total = time.perf_counter()
    results = batch_func(texts, batch_size=batch_size)
    elapsed_total = (time.perf_counter() - start_total) * 1000

    for i, result in enumerate(results):
        is_safe = result.get("safe", True)
        y_pred.append(0 if is_safe else 1)
        scores.append(result.get("score", 0.0))
        latencies.append(elapsed_total / len(texts))

    metrics = _calculate_metrics(y_true, y_pred, latencies)
    metrics["dataset"] = name
    metrics["avg_score"] = round(np.mean(scores), 4) if scores else 0.0
    metrics["samples"] = len(texts)
    return metrics


def evaluate_detector(detector_func, data, name="dataset", sample_size=None):
    """
    Evaluate a detector against a dataset.

    Args:
        detector_func: Callable[[str], {"safe": bool}]
        data: DataFrame with 'text' and 'label' columns
        name: Dataset name for reporting
        sample_size: Optional sample limit

    Returns: dict of metrics
    """
    if sample_size and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42)

    texts = data["text"].tolist()
    y_true = data["label"].tolist()
    y_pred = []
    scores = []
    latencies = []

    for text in texts:
        start = time.perf_counter()
        result = detector_func(text)
        elapsed = (time.perf_counter() - start) * 1000

        is_safe = result.get("safe", True)
        y_pred.append(0 if is_safe else 1)
        scores.append(result.get("score", 0.0))
        latencies.append(elapsed)

    metrics = _calculate_metrics(y_true, y_pred, latencies)
    metrics["dataset"] = name
    metrics["avg_score"] = round(np.mean(scores), 4) if scores else 0.0
    metrics["samples"] = len(texts)
    return metrics


def compare_detectors(detectors, datasets):
    """
    Compare multiple detectors across multiple datasets.

    Args:
        detectors: dict of name -> (detector_func OR (detector_func, batch_func))
        datasets: dict of name -> DataFrame

    Returns: pd.DataFrame with detectors as rows, metrics as columns
    """
    import sys
    total = len(detectors) * len(datasets)
    done = 0
    rows = []
    for det_name, det_entry in detectors.items():
        if isinstance(det_entry, tuple):
            det_func, batch_func = det_entry
        else:
            det_func = det_entry
            batch_func = None

        for ds_name, ds_data in datasets.items():
            done += 1
            print(f"  [{done}/{total}] {det_name} on {ds_name} ({len(ds_data)} samples)...", flush=True)
            start_t = time.perf_counter()
            if batch_func:
                metrics = _batch_evaluate_detector(batch_func, ds_data, name=ds_name)
            else:
                metrics = evaluate_detector(det_func, ds_data, name=ds_name)
            elapsed = time.perf_counter() - start_t
            print(f"    -> acc={metrics['accuracy']*100:.1f}% ({elapsed:.1f}s)", flush=True)
            metrics["detector"] = det_name
            rows.append(metrics)

    df = pd.DataFrame(rows)
    return df


def _make_detectors():
    """Build the standard set of detectors for comparison."""
    pattern_agent = ThreatAgent()

    def pattern_detector(text):
        return pattern_agent.process(text)

    ml_classifier = MLPromptClassifier()
    try:
        ml_classifier.load_model("ml_classifier.pkl")
    except Exception:
        pass

    def ml_detector(text):
        return ml_classifier.predict(text)

    def hybrid_detector(text):
        pattern_result = pattern_agent.process(text)
        if not pattern_result["safe"]:
            return pattern_result
        if ml_classifier.is_trained:
            ml_result = ml_classifier.predict(text)
            if not ml_result["safe"]:
                return ml_result
        return {"safe": True, "score": 0.0}

    detectors = {
        "Pattern-Based": pattern_detector,
    }
    if ml_classifier.is_trained:
        detectors["ML (TF-IDF+NB)"] = ml_detector
        detectors["Hybrid (Pattern+ML)"] = hybrid_detector

    transformer_clf = TransformerClassifier()
    try:
        transformer_clf.load_model("transformer_classifier")
    except Exception:
        pass

    def transformer_detector(text):
        return transformer_clf.predict(text)

    def transformer_batch(texts, batch_size=64):
        return transformer_clf.predict_batch(texts, batch_size=batch_size)

    if transformer_clf.is_trained:
        detectors["Transformer (DistilBERT)"] = (transformer_detector, transformer_batch)

    return detectors


def generate_report(results_df):
    """Generate a formatted text report from comparison results."""
    lines = []
    lines.append("=" * 80)
    lines.append("UNIFIED EVALUATION REPORT")
    lines.append("=" * 80)

    for detector in results_df["detector"].unique():
        det_data = results_df[results_df["detector"] == detector]
        lines.append(f"\n{'─' * 80}")
        lines.append(f"  {detector}")
        lines.append(f"{'─' * 80}")
        lines.append(f"{'Dataset':<25} {'Samples':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Spec':<8}")
        lines.append("-" * 73)
        for _, row in det_data.iterrows():
            lines.append(
                f"{row['dataset']:<25} {row['samples']:<8} "
                f"{row['accuracy']*100:>5.1f}%{'':<2} "
                f"{row['precision']*100:>5.1f}%{'':<2} "
                f"{row['recall']*100:>5.1f}%{'':<2} "
                f"{row['f1']*100:>5.1f}%{'':<2} "
                f"{row['specificity']*100:>5.1f}%"
            )

        avg = det_data[["accuracy", "precision", "recall", "f1", "specificity"]].mean()
        lines.append("-" * 73)
        lines.append(
            f"{'AVERAGE':<25} {'':8} "
            f"{avg['accuracy']*100:>5.1f}%{'':<2} "
            f"{avg['precision']*100:>5.1f}%{'':<2} "
            f"{avg['recall']*100:>5.1f}%{'':<2} "
            f"{avg['f1']*100:>5.1f}%{'':<2} "
            f"{avg['specificity']*100:>5.1f}%"
        )

    lines.append(f"\n{'=' * 80}")
    return "\n".join(lines)


def evaluate_multi_turn(detectors):
    """Evaluate context-aware vs isolated detection on multi-turn scenarios."""
    print("\n" + "=" * 80)
    print("MULTI-TURN ATTACK DETECTION EVALUATION")
    print("=" * 80)

    _, multi_df, single_df = generate_multi_turn_dataset(
        n_clean=10, n_direct=10, n_delayed=10, n_distributed=10, n_context=10
    )

    conversations = []
    labels = []
    for _, r in multi_df.iterrows():
        turns = [line.replace("User: ", "") for line in r["text"].split("\n") if line.startswith("User:")]
        conversations.append(turns)
        labels.append(r["label"])

    total_conv = len(conversations)
    print(f"\n{total_conv} conversations ({sum(labels)} attacks, {total_conv - sum(labels)} clean)")
    print(f"  Turns per conversation: {multi_df['num_turns'].describe()['mean']:.0f} avg\n")

    rows = []
    for det_name, det_entry in detectors.items():
        if isinstance(det_entry, tuple):
            det_func, _ = det_entry
        else:
            det_func = det_entry

        ctx = ContextDetector(det_func, context_window=3, strategy="concat", name=f"{det_name}+Ctx")

        iso_correct = 0
        ctx_correct = 0
        for conv, label in zip(conversations, labels):
            ctx.reset()
            iso_results = [det_func(msg) for msg in conv]
            ctx_results = [ctx.predict(msg) for msg in conv]
            iso_attack = any(not r["safe"] for r in iso_results)
            ctx_attack = any(not r["safe"] for r in ctx_results)
            if iso_attack == (label == 1):
                iso_correct += 1
            if ctx_attack == (label == 1):
                ctx_correct += 1

        iso_acc = iso_correct / total_conv
        ctx_acc = ctx_correct / total_conv
        row = {
            "detector": det_name,
            "isolated_acc": f"{iso_acc*100:.1f}%",
            "context_acc": f"{ctx_acc*100:.1f}%",
            "isolated_correct": iso_correct,
            "context_correct": ctx_correct,
            "total": total_conv,
        }
        rows.append(row)
        print(f"  {det_name:.<35} isolated={iso_acc*100:.1f}%  context={ctx_acc*100:.1f}%")

    results_df = pd.DataFrame(rows)
    results_df.to_csv("multi_turn_evaluation_results.csv", index=False)
    print(f"\nResults saved to multi_turn_evaluation_results.csv")
    return results_df


def main():
    print("=" * 80)
    print("UNIFIED EVALUATION: Multi-Detector Comparison")
    print("=" * 80)

    loader = DatasetLoader()
    detectors = _make_detectors()

    print(f"\nLoaded {len(detectors)} detector(s): {', '.join(detectors.keys())}")

    datasets = loader.get_benchmark_data()
    dataset_names = [n for n in ["command_injection", "harmful_behaviors", "harmful_strings", "transfer_behaviors", "jailbreak_bench", "shieldlm_pi", "neuralchemy_pi"] if n in datasets]
    datasets_ordered = {n: datasets[n] for n in dataset_names}
    # Sample large datasets for speed
    for name in datasets_ordered:
        if len(datasets_ordered[name]) > 5000:
            datasets_ordered[name] = datasets_ordered[name].sample(n=5000, random_state=42)
    print(f"Loaded {len(datasets_ordered)} dataset(s): {', '.join(datasets_ordered.keys())}")
    total_samples = sum(len(d) for d in datasets_ordered.values())
    print(f"Total evaluation samples: {total_samples}")

    results = compare_detectors(detectors, datasets_ordered)

    print(generate_report(results))

    csv_path = "unified_evaluation_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    evaluate_multi_turn(detectors)

    return results


if __name__ == "__main__":
    main()
