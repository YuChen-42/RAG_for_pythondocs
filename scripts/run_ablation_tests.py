import json
import os
import subprocess
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src import config
from src.evaluator import Evaluator
from src.rag_engine import RAGEngine


def load_qa_pairs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(test_name: str, use_rewrite: bool, use_rerank: bool, output_dir: str, report_suffix: str = None, engine: RAGEngine = None, evaluator: Evaluator = None):
    console = Console()
    console.print(f"\n[bold green]===== {test_name} =====[/bold green]")
    console.print(f"Query Rewrite: {'ON' if use_rewrite else 'OFF'}, Rerank: {'ON' if use_rerank else 'OFF'}")

    qa_pairs = load_qa_pairs(config.QA_PAIRS_PATH)
    if engine is None:
        engine = RAGEngine()
    if evaluator is None:
        evaluator = Evaluator()

    enriched_pairs = []
    for qa in tqdm(qa_pairs, desc="生成答案"):
        result = engine.query(qa["question"], use_query_rewrite=use_rewrite, use_rerank=use_rerank)
        enriched_pairs.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [r["text"] for r in engine.retriever.retrieve(qa["question"])],
            "ground_truth": qa["ground_truth"],
        })

    results = evaluator.evaluate(enriched_pairs)

    table = Table(title=f"Ragas 评估结果 - {test_name}")
    table.add_column("问题", style="cyan")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Answer Relevancy", justify="right")
    table.add_column("Context Recall", justify="right")

    for i, r in enumerate(results):
        table.add_row(
            enriched_pairs[i]["question"][:40] + "...",
            f"{r.get('faithfulness', 0):.3f}",
            f"{r.get('answer_relevancy', 0):.3f}",
            f"{r.get('context_recall', 0):.3f}",
        )

    console.print(table)

    os.makedirs(output_dir, exist_ok=True)
    suffix = report_suffix if report_suffix else ('on' if (use_rewrite or use_rerank) else 'off')
    path = os.path.join(output_dir, f"ragas_report_{suffix}.json")
    evaluator.save_report(results, path)
    console.print(f"[bold blue]报告已保存: {path}[/bold blue]")

    return results


def run_comparison(test_name: str, on_config: dict, off_config: dict, output_dir: str, result_cache: dict, engine: RAGEngine, evaluator: Evaluator):
    console = Console()
    console.print(f"\n[bold yellow]>>> 运行 {test_name} 对比实验[/bold yellow]")

    on_key = (on_config["use_rewrite"], on_config["use_rerank"])
    off_key = (off_config["use_rewrite"], off_config["use_rerank"])

    if on_key in result_cache:
        console.print(f"[dim]使用缓存结果: {test_name} - ON[/dim]")
        on_results = result_cache[on_key]
    else:
        on_results = run_experiment(f"{test_name} - ON", **on_config, output_dir=output_dir, report_suffix="on", engine=engine, evaluator=evaluator)
        result_cache[on_key] = on_results

    if off_key in result_cache:
        console.print(f"[dim]使用缓存结果: {test_name} - OFF[/dim]")
        off_results = result_cache[off_key]
    else:
        off_results = run_experiment(f"{test_name} - OFF", **off_config, output_dir=output_dir, report_suffix="off", engine=engine, evaluator=evaluator)
        result_cache[off_key] = off_results

    # Generate comparison chart
    on_path = os.path.join(output_dir, "ragas_report_on.json")
    off_path = os.path.join(output_dir, "ragas_report_off.json")
    chart_path = os.path.join(output_dir, "comparison_chart.png")

    subprocess.run([
        "python", "scripts/generate_comparison_chart.py",
        "--baseline", off_path,
        "--optimized", on_path,
        "--output", chart_path,
        "--baseline-label", "OFF",
        "--optimized-label", "ON",
    ], check=True)

    console.print(f"[bold blue]对比图已保存: {chart_path}[/bold blue]")

    # Summary
    metrics = ["faithfulness", "answer_relevancy", "context_recall"]
    on_avg = {m: sum(r.get(m, 0) for r in on_results) / len(on_results) for m in metrics}
    off_avg = {m: sum(r.get(m, 0) for r in off_results) / len(off_results) for m in metrics}

    table = Table(title=f"{test_name} 平均指标对比")
    table.add_column("指标", style="cyan")
    table.add_column("ON", justify="right")
    table.add_column("OFF", justify="right")
    table.add_column("提升", justify="right")

    for m in metrics:
        delta = on_avg[m] - off_avg[m]
        table.add_row(m, f"{on_avg[m]:.3f}", f"{off_avg[m]:.3f}", f"{delta:+.3f}")

    console.print(table)


def main():
    console = Console()
    console.print("[bold magenta]开始 RAG 消融实验[/bold magenta]")

    engine = RAGEngine()
    evaluator = Evaluator()
    result_cache = {}

    # Test 1: Query Rewrite
    run_comparison(
        "Test 1: Query Rewrite",
        on_config={"use_rewrite": True, "use_rerank": config.USE_RERANK},
        off_config={"use_rewrite": False, "use_rerank": config.USE_RERANK},
        output_dir="results/test1",
        result_cache=result_cache,
        engine=engine,
        evaluator=evaluator,
    )

    # Test 2: Rerank
    run_comparison(
        "Test 2: Rerank",
        on_config={"use_rewrite": config.USE_QUERY_REWRITE, "use_rerank": True},
        off_config={"use_rewrite": config.USE_QUERY_REWRITE, "use_rerank": False},
        output_dir="results/test2",
        result_cache=result_cache,
        engine=engine,
        evaluator=evaluator,
    )

    # Test 3: Combined
    run_comparison(
        "Test 3: Query Rewrite + Rerank",
        on_config={"use_rewrite": True, "use_rerank": True},
        off_config={"use_rewrite": False, "use_rerank": False},
        output_dir="results/test3",
        result_cache=result_cache,
        engine=engine,
        evaluator=evaluator,
    )

    console.print("\n[bold green]所有消融实验已完成！[/bold green]")


if __name__ == "__main__":
    main()
