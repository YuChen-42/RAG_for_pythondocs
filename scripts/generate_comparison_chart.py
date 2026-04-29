import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="生成 Ragas 指标对比图")
    parser.add_argument("--baseline", required=True, help="基线评估报告 JSON 路径")
    parser.add_argument("--optimized", required=True, help="优化评估报告 JSON 路径")
    parser.add_argument("--output", required=True, help="输出图片路径")
    parser.add_argument("--baseline-label", default="Baseline", help="基线标签")
    parser.add_argument("--optimized-label", default="Optimized", help="优化标签")
    args = parser.parse_args()

    with open(args.baseline, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.optimized, "r", encoding="utf-8") as f:
        optimized = json.load(f)

    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    baseline_avg = {m: sum(r.get(m, 0) for r in baseline) / len(baseline) for m in metrics}
    optimized_avg = {m: sum(r.get(m, 0) for r in optimized) / len(optimized) for m in metrics}

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], [baseline_avg[m] for m in metrics], width, label=args.baseline_label)
    ax.bar([i + width/2 for i in x], [optimized_avg[m] for m in metrics], width, label=args.optimized_label)
    ax.set_ylabel("Score")
    ax.set_title("Ragas Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"对比图已保存到 {args.output}")


if __name__ == "__main__":
    main()
