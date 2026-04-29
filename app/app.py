from flask import Flask, render_template, request, jsonify, Response
import json
from src.rag_engine import RAGEngine

app = Flask(__name__, template_folder="templates")
engine = RAGEngine()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json()
    query = data.get("query", "")
    use_rewrite = data.get("use_query_rewrite", True)
    if not query:
        return jsonify({"error": "query is required"}), 400
    result = engine.query(query, use_query_rewrite=use_rewrite)
    return jsonify({"answer": result["answer"], "sources": result["sources"]})


@app.route("/api/stream")
def api_stream():
    query = request.args.get("q", "")
    use_rewrite = request.args.get("rewrite", "1") == "1"
    if not query:
        return jsonify({"error": "q is required"}), 400

    def generate():
        result = engine.query(query, stream=True, use_query_rewrite=use_rewrite)
        for token in result["stream"]:
            yield f"data: {token}\n\n"
        sources_json = json.dumps(result["sources"], ensure_ascii=False)
        yield f"data: [SOURCES]{sources_json}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
