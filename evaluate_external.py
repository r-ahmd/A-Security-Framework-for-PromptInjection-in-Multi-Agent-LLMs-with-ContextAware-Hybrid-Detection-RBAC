"""
External Dataset Evaluation Script
Tests threat detection against independent datasets
"""

import pandas as pd
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.threat_agent import ThreatAgent

def evaluate_command_injection():
    """Evaluate against CommandInjection dataset (2627 samples)"""
    print("\n" + "="*60)
    print("EVALUATION: CommandInjection Dataset")
    print("="*60)
    
    df = pd.read_csv("PromptInjection-to-CommandInjection - CommandInjection.csv")
    print(f"Total samples: {len(df)}")
    print(f"Label distribution: {df['Label'].value_counts().to_dict()}")
    
    agent = ThreatAgent()
    
    tp = fp = tn = fn = 0
    results = []
    
    for _, row in df.iterrows():
        text = str(row['Prompt'])
        true_label = row['Label']  # 0=clean, 1=injection
        
        result = agent.process(text)
        predicted = 0 if result['safe'] else 1
        
        if true_label == 1 and predicted == 1:
            tp += 1
        elif true_label == 0 and predicted == 0:
            tn += 1
        elif true_label == 0 and predicted == 1:
            fp += 1
        else:
            fn += 1
        
        results.append({
            'text': text[:100],
            'true_label': true_label,
            'predicted': predicted,
            'score': result['score']
        })
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nResults:")
    print(f"  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    return {
        'dataset': 'CommandInjection',
        'total': total,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_harmful_behaviors():
    """Evaluate against harmful_behaviors dataset (521 samples)"""
    print("\n" + "="*60)
    print("EVALUATION: Harmful Behaviors Dataset")
    print("="*60)
    
    df = pd.read_csv("harmful_behaviors.csv")
    print(f"Total samples: {len(df)}")
    
    agent = ThreatAgent()
    
    detected = 0
    missed = 0
    scores = []
    
    for _, row in df.iterrows():
        text = str(row['goal'])
        result = agent.process(text)
        scores.append(result['score'])
        
        if not result['safe']:
            detected += 1
        else:
            missed += 1
    
    detection_rate = detected / len(df) if len(df) > 0 else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print(f"\nResults:")
    print(f"  Detected:   {detected}/{len(df)}")
    print(f"  Missed:     {missed}/{len(df)}")
    print(f"  Detection Rate: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
    print(f"  Average Score:  {avg_score:.4f}")
    
    # Show some missed examples
    if missed > 0:
        print(f"\nSample Missed Attacks (first 5):")
        count = 0
        for _, row in df.iterrows():
            result = agent.process(str(row['goal']))
            if result['safe'] and count < 5:
                print(f"  - {row['goal'][:80]}...")
                count += 1
    
    return {
        'dataset': 'Harmful Behaviors',
        'total': len(df),
        'detected': detected,
        'missed': missed,
        'detection_rate': detection_rate,
        'avg_score': avg_score
    }

def evaluate_harmful_strings():
    """Evaluate against harmful_strings dataset (575 samples)"""
    print("\n" + "="*60)
    print("EVALUATION: Harmful Strings Dataset")
    print("="*60)
    
    df = pd.read_csv("harmful_strings.csv")
    print(f"Total samples: {len(df)}")
    
    agent = ThreatAgent()
    
    detected = 0
    missed = 0
    scores = []
    
    for _, row in df.iterrows():
        text = str(row['target'])
        result = agent.process(text)
        scores.append(result['score'])
        
        if not result['safe']:
            detected += 1
        else:
            missed += 1
    
    detection_rate = detected / len(df) if len(df) > 0 else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print(f"\nResults:")
    print(f"  Detected:   {detected}/{len(df)}")
    print(f"  Missed:     {missed}/{len(df)}")
    print(f"  Detection Rate: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
    print(f"  Average Score:  {avg_score:.4f}")
    
    return {
        'dataset': 'Harmful Strings',
        'total': len(df),
        'detected': detected,
        'missed': missed,
        'detection_rate': detection_rate,
        'avg_score': avg_score
    }

def evaluate_transfer_behaviors():
    """Evaluate against transfer_expriment_behaviors dataset"""
    print("\n" + "="*60)
    print("EVALUATION: Transfer Experiment Behaviors Dataset")
    print("="*60)
    
    try:
        df = pd.read_csv("transfer_expriment_behaviors.csv", on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading file: {e}")
        lines = []
        with open("transfer_expriment_behaviors.csv", 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and line != 'goals':
                    lines.append(line)
        df = pd.DataFrame({'goals': lines})
    
    print(f"Total samples: {len(df)}")
    
    agent = ThreatAgent()
    
    detected = 0
    missed = 0
    scores = []
    
    for _, row in df.iterrows():
        text = str(row.iloc[0])  # First column
        result = agent.process(text)
        scores.append(result['score'])
        
        if not result['safe']:
            detected += 1
        else:
            missed += 1
    
    detection_rate = detected / len(df) if len(df) > 0 else 0
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print(f"\nResults:")
    print(f"  Detected:   {detected}/{len(df)}")
    print(f"  Missed:     {missed}/{len(df)}")
    print(f"  Detection Rate: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
    print(f"  Average Score:  {avg_score:.4f}")
    
    return {
        'dataset': 'Transfer Behaviors',
        'total': len(df),
        'detected': detected,
        'missed': missed,
        'detection_rate': detection_rate,
        'avg_score': avg_score
    }

def main():
    print("="*60)
    print("EXTERNAL DATASET EVALUATION")
    print("Threat Detection Agent Performance")
    print("="*60)
    
    results = []
    
    # Run evaluations
    results.append(evaluate_command_injection())
    results.append(evaluate_harmful_behaviors())
    results.append(evaluate_harmful_strings())
    results.append(evaluate_transfer_behaviors())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY ACROSS ALL DATASETS")
    print("="*60)
    
    total_samples = sum(r['total'] for r in results)
    total_detected = sum(r.get('detected', r.get('tp', 0)) for r in results)
    
    print(f"\n{'Dataset':<25} {'Samples':<10} {'Detected':<10} {'Rate':<10}")
    print("-"*55)
    for r in results:
        detected = r.get('detected', r.get('tp', 0))
        rate = r.get('detection_rate', r.get('accuracy', 0))
        print(f"{r['dataset']:<25} {r['total']:<10} {detected:<10} {rate*100:.1f}%")
    
    print("-"*55)
    overall_rate = total_detected / total_samples if total_samples > 0 else 0
    print(f"{'TOTAL':<25} {total_samples:<10} {total_detected:<10} {overall_rate*100:.1f}%")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("evaluation_results.csv", index=False)
    print(f"\nResults saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
