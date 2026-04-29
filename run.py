import argparse


def main():
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--build-index", action="store_true", help="构建索引")
    parser.add_argument("--query", type=str, help="CLI 问答")
    parser.add_argument("--no-query-rewrite", action="store_true", help="禁用 Query 改写（用于对比效果）")
    parser.add_argument("--web", action="store_true", help="启动 Web 服务")
    parser.add_argument("--port", type=int, default=5000, help="Web 服务端口（默认 5000）")
    parser.add_argument("--eval", action="store_true", help="运行 Ragas 评估")
    parser.add_argument("--config", type=str, default="baseline", choices=["baseline", "optimized"], help="评估配置")
    parser.add_argument("--ablation", action="store_true", help="运行三组消融实验（Query Rewrite / Rerank / 联合）")
    args = parser.parse_args()

    if args.build_index:
        from scripts.build_index import main as build
        build()
    elif args.query:
        from src.rag_engine import RAGEngine
        engine = RAGEngine()
        result = engine.query(args.query, use_query_rewrite=not args.no_query_rewrite)
        print(f"\n答案:\n{result['answer']}\n")
        print(f"来源: {', '.join(result['sources'])}")
    elif args.web:
        from app.app import app
        app.run(host="0.0.0.0", port=args.port, debug=True)
    elif args.eval:
        from scripts.evaluate import main as evaluate
        evaluate(args.config)
    elif args.ablation:
        from scripts.run_ablation_tests import main as run_ablation
        run_ablation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
