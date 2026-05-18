import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"⚠️ File not found: {filepath}")
    return {}

def main():
    file1 = "/home/c/my_apr_project/eval_results/eval_20260411_221134.json"
    file2 = "/home/c/my_apr_project/eval_results/eval_20260413_140126.json" 
    
    data1 = load_json(file1)
    data2 = load_json(file2)
    
    all_results = {}
    all_results.update(data1)
    all_results.update(data2)
    
    summary = []
    for mode, content in all_results.items():
        results = content.get("results", [])
        if not results:
            continue
            
        total_bugs = len(results)
        repaired = sum(1 for r in results if r.get("repaired"))
        repair_rate = (repaired / total_bugs) * 100 if total_bugs > 0 else 0
        avg_time = sum(r.get("time_seconds", 0) for r in results) / total_bugs if total_bugs > 0 else 0
        avg_attempts = sum(r.get("attempts_used", 0) for r in results) / total_bugs if total_bugs > 0 else 0
        
        summary.append({
            "Mode": mode,
            "Total Bugs": total_bugs,
            "Fixed Bugs": repaired,
            "Repair Rate (%)": round(repair_rate, 2),
            "Avg Time (s)": round(avg_time, 2),
            "Avg Attempts": round(avg_attempts, 2)
        })
        
    df = pd.DataFrame(summary)
    df = df.sort_values(by="Repair Rate (%)", ascending=True)
    
    print("\n" + "="*60)
    print("📊 Comparison Summary")
    print("="*60)
    print(df.to_markdown(index=False))
    print("="*60 + "\n")
    
    df.to_csv("evaluation_comparison.csv", index=False)
    print("✅ Saved tabular data to 'evaluation_comparison.csv'")
    
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999' if 'base' in m else '#66b3ff' for m in df['Mode']]
    
    bars = plt.barh(df['Mode'], df['Repair Rate (%)'], color=colors)
    plt.xlabel('Repair Success Rate (%)', fontsize=12, fontweight='bold')
    plt.title('TestMate: Base Model vs. Finetuned Model Performance', fontsize=14, fontweight='bold')
    plt.xlim(0, 100)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                 f'{width}%', ha='left', va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("repair_rate_comparison.png", dpi=300)
    plt.show()
    print("✅ Saved chart to 'repair_rate_comparison.png'")

if __name__ == "__main__":
    main()