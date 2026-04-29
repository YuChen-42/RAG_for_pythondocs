import json
import argparse
import os
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from src import config
from src.evaluator import Evaluator
from src.rag_engine import RAGEngine


def load_qa_pairs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_qa_and_evaluate(config_name: str = "baseline"):
    console = Console()

    if config_name == "optimized":
        config.CHUNK_SIZE = 256
        config.CHUNK_OVERLAP = 50
        config.USE_RERANK = True
    else:
        config.CHUNK_SIZE = 512
        config.CHUNK_OVERLAP = 100
        config.USE_RERANK = False

    qa_pairs = load_qa_pairs(config.QA_PAIRS_PATH)
    engine = RAGEngine()
    evaluator = Evaluator()

    console.print(f"[bold green]开始评估: {config_name}[/bold green]")

    enriched_pairs = []
    for qa in tqdm(qa_pairs, desc="生成答案"):
        result = engine.query(qa["question"])
        enriched_pairs.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [r["text"] for r in engine.retriever.retrieve(qa["question"])],
            "ground_truth": qa["ground_truth"],
        })

    results = evaluator.evaluate(enriched_pairs)

    table = Table(title=f"Ragas 评估结果 - {config_name}")
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

    os.makedirs("results", exist_ok=True)
    path = f"results/ragas_report_{config_name}.json"
    evaluator.save_report(results, path)
    console.print(f"[bold blue]报告已保存: {path}[/bold blue]")

    return results


def main(config_name: str = "baseline"):
    run_qa_and_evaluate(config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline", choices=["baseline", "optimized"])
    args = parser.parse_args()
    main(args.config)
