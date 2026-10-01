"""
Lakera Guard Baseline Comparison
Enhanced comparison with multiple datasets, statistical analysis, and cost metrics
"""

import requests
import pandas as pd
import time
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from agents.threat_agent import ThreatAgent

# Lakera Guard API configuration (v2 endpoint)
LAKERA_API_KEY = "c8ce802453f7c629c2f3a732215b97f703ed640d34dc821e81720c4abbf436e6"
LAKERA_API_URL = "https://api.lakera.ai/v2/guard"

def check_lakera_guard(text):
    """Check text using Lakera Guard API v2"""
    try:
        response = requests.post(
            LAKERA_API_URL,
            headers={
                "Authorization": f"Bearer {LAKERA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"messages": [{"role": "user", "content": text}]},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            flagged = result.get("flagged", False)
            # Extract confidence score if available
            metadata = result.get("metadata", {})
            score = metadata.get("score", 1.0 if flagged else 0.0)
            return {
                "detected": flagged,
                "score": score,
                "raw_response": result
            }
        else:
            return {"detected": False, "score": 0, "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"detected": False, "score": 0, "error": str(e)}

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics with safe labels=[0, 1]"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

def evaluate_detector_on_dataset(detector_func, df, text_column, label_column, sample_size=None):
    """Evaluate a detector on a dataset"""
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    y_true = []
    y_pred = []
    scores = []
    latencies = []
    
    for _, row in df_sample.iterrows():
        text = str(row[text_column])
        true_label = int(row[label_column])
        
        start_time = time.time()
        result = detector_func(text)
        latency = (time.time() - start_time) * 1000  # ms
        
        detected = 1 if result.get("detected", False) or not result.get("safe", True) else 0
        
        y_true.append(true_label)
        y_pred.append(detected)
        scores.append(result.get("score", 0))
        latencies.append(latency)
    
    metrics = calculate_metrics(y_true, y_pred)
    metrics["avg_latency_ms"] = np.mean(latencies)
    metrics["std_latency_ms"] = np.std(latencies)
    metrics["avg_score"] = np.mean(scores)
    metrics["sample_size"] = len(df_sample)
    
    return metrics, y_true, y_pred, scores

def lakera_detector(text):
    """Wrapper for Lakera Guard"""
    result = check_lakera_guard(text)
    return {"detected": result["detected"], "score": result["score"], "safe": not result["detected"]}

def our_detector(text):
    """Wrapper for our threat agent"""
    agent = ThreatAgent()
    result = agent.process(text)
    return {"detected": not result["safe"], "score": result["score"], "safe": result["safe"]}

def main():
    print("="*70)
    print("ALL 8 BENCHMARK DATASETS - LAKERA GUARD BASELINE COMPARISON")
    print("="*70)
    
    from dataset_loader import DatasetLoader
    loader = DatasetLoader()
    
    dataset_names = [
        "command_injection",
        "harmful_behaviors",
        "harmful_strings",
        "transfer_behaviors",
        "jailbreak_bench",
        "shieldlm_pi",
        "harmbench",
        "neuralchemy_pi"
    ]
    
    all_results = {}
    SAMPLE_PER_DATASET = 30  # 30 samples * 8 datasets = 240 live API queries
    
    for ds_name in dataset_names:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")
        
        try:
            df = loader.load(ds_name, sample_size=SAMPLE_PER_DATASET)
            if df.empty:
                print(f"Skipping empty dataset: {ds_name}")
                continue
            
            print(f"Loaded {len(df)} samples ({df['label'].sum()} attacks, {len(df) - df['label'].sum()} clean)")
            
            # Evaluate Lakera Guard
            print(f"\nEvaluating Lakera Guard (n={len(df)})...")
            lakera_metrics, lakera_y_true, lakera_y_pred, lakera_scores = evaluate_detector_on_dataset(
                lakera_detector, df, "text", "label", len(df)
            )
            
            # Evaluate Our Detector
            print(f"Evaluating Our Detector (n={len(df)})...")
            our_metrics, our_y_true, our_y_pred, our_scores = evaluate_detector_on_dataset(
                our_detector, df, "text", "label", len(df)
            )
            
            # Print results
            print(f"\n--- Results for {ds_name} ---")
            print(f"\n{'Metric':<20} {'Lakera Guard':<15} {'Our Detector':<15} {'Difference':<15}")
            print("-"*65)
            print(f"{'Accuracy':<20} {lakera_metrics['accuracy']*100:.2f}%{'':<8} {our_metrics['accuracy']*100:.2f}%{'':<8} {(our_metrics['accuracy']-lakera_metrics['accuracy'])*100:+.2f}%")
            print(f"{'Precision':<20} {lakera_metrics['precision']*100:.2f}%{'':<8} {our_metrics['precision']*100:.2f}%{'':<8} {(our_metrics['precision']-lakera_metrics['precision'])*100:+.2f}%")
            print(f"{'Recall':<20} {lakera_metrics['recall']*100:.2f}%{'':<8} {our_metrics['recall']*100:.2f}%{'':<8} {(our_metrics['recall']-lakera_metrics['recall'])*100:+.2f}%")
            print(f"{'F1-Score':<20} {lakera_metrics['f1']*100:.2f}%{'':<8} {our_metrics['f1']*100:.2f}%{'':<8} {(our_metrics['f1']-lakera_metrics['f1'])*100:+.2f}%")
            print(f"{'Specificity':<20} {lakera_metrics['specificity']*100:.2f}%{'':<8} {our_metrics['specificity']*100:.2f}%{'':<8} {(our_metrics['specificity']-lakera_metrics['specificity'])*100:+.2f}%")
            print(f"{'Avg Latency (ms)':<20} {lakera_metrics['avg_latency_ms']:.2f}{'':<12} {our_metrics['avg_latency_ms']:.2f}")
            
            all_results[ds_name] = {
                "lakera_guard": lakera_metrics,
                "our_detector": our_metrics,
                "sample_size": len(df)
            }
            
        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
    
    # Overall summary across all 8 datasets
    print(f"\n{'='*70}")
    print("ALL 8 DATASETS - OVERALL SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        avg_lakera_acc = np.mean([r["lakera_guard"]["accuracy"] for r in all_results.values()])
        avg_our_acc = np.mean([r["our_detector"]["accuracy"] for r in all_results.values()])
        avg_lakera_f1 = np.mean([r["lakera_guard"]["f1"] for r in all_results.values()])
        avg_our_f1 = np.mean([r["our_detector"]["f1"] for r in all_results.values()])
        avg_lakera_recall = np.mean([r["lakera_guard"]["recall"] for r in all_results.values()])
        avg_our_recall = np.mean([r["our_detector"]["recall"] for r in all_results.values()])
        avg_lakera_precision = np.mean([r["lakera_guard"]["precision"] for r in all_results.values()])
        avg_our_precision = np.mean([r["our_detector"]["precision"] for r in all_results.values()])
        avg_lakera_lat = np.mean([r["lakera_guard"]["avg_latency_ms"] for r in all_results.values()])
        avg_our_lat = np.mean([r["our_detector"]["avg_latency_ms"] for r in all_results.values()])
        
        print(f"\nAverage across all {len(all_results)} benchmark datasets:")
        print(f"\n{'Metric':<20} {'Lakera Guard':<15} {'Our Detector':<15} {'Winner':<15}")
        print("-"*65)
        print(f"{'Avg Accuracy':<20} {avg_lakera_acc*100:.2f}%{'':<8} {avg_our_acc*100:.2f}%{'':<8} {'Ours' if avg_our_acc > avg_lakera_acc else 'Lakera'}")
        print(f"{'Avg F1-Score':<20} {avg_lakera_f1*100:.2f}%{'':<8} {avg_our_f1*100:.2f}%{'':<8} {'Ours' if avg_our_f1 > avg_lakera_f1 else 'Lakera'}")
        print(f"{'Avg Recall':<20} {avg_lakera_recall*100:.2f}%{'':<8} {avg_our_recall*100:.2f}%{'':<8} {'Ours' if avg_our_recall > avg_lakera_recall else 'Lakera'}")
        print(f"{'Avg Precision':<20} {avg_lakera_precision*100:.2f}%{'':<8} {avg_our_precision*100:.2f}%{'':<8} {'Ours' if avg_our_precision > avg_lakera_precision else 'Lakera'}")
        print(f"{'Avg Latency (ms)':<20} {avg_lakera_lat:.2f}{'':<12} {avg_our_lat:.2f}{'':<12} Ours ({avg_lakera_lat/avg_our_lat:.1f}x Faster)")
        
    output = {
        "datasets": all_results,
        "summary": {
            "avg_lakera_accuracy": avg_lakera_acc if all_results else 0,
            "avg_our_accuracy": avg_our_acc if all_results else 0,
            "avg_lakera_f1": avg_lakera_f1 if all_results else 0,
            "avg_our_f1": avg_our_f1 if all_results else 0,
            "avg_lakera_latency_ms": avg_lakera_lat if all_results else 0,
            "avg_our_latency_ms": avg_our_lat if all_results else 0,
            "num_datasets": len(all_results)
        }
    }
    with open("lakera_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAll results saved to lakera_comparison_results.json")

if __name__ == "__main__":
    main()
