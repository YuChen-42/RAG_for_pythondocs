import json
import typing as t
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms.base import BaseRagasLLM, LLMResult
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import ChatGeneration
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from datasets import Dataset
from src.llm_client import LLMClient
from src.embedding import EmbeddingModel


class RagasLLMWrapper(BaseRagasLLM):
    """Wrap our LLMClient to satisfy Ragas BaseRagasLLM interface."""

    def __init__(self, client: LLMClient):
        super().__init__()
        self._client = client

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        messages = []
        for m in prompt.to_messages():
            role = "user"
            if hasattr(m, "type"):
                role = m.type
            if role in ("system", "human"):
                role = "user" if role == "human" else "system"
            messages.append({"role": role, "content": m.content})

        response = self._client.chat(messages, temperature=temperature, stream=False)
        content = response.choices[0].message.content or ""

        msg = AIMessage(content=content)
        cg = ChatGeneration(
            message=msg,
            generation_info={"finish_reason": "stop"},
        )
        return LLMResult(generations=[[cg]], llm_output={})

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: t.Optional[float] = 0.01,
        stop: t.Optional[t.List[str]] = None,
        callbacks=None,
    ) -> LLMResult:
        return self.generate_text(prompt, n, temperature or 0.01, stop, callbacks)

    def is_finished(self, response: LLMResult) -> bool:
        try:
            return bool(response.generations[0][0].message.content)
        except Exception:
            return False


class RagasEmbeddingWrapper(BaseRagasEmbeddings):
    """Wrap our local EmbeddingModel to satisfy Ragas BaseRagasEmbeddings interface."""

    def __init__(self):
        super().__init__()
        self._model = EmbeddingModel()

    def embed_query(self, text: str) -> t.List[float]:
        return self._model.encode([text])[0].tolist()

    def embed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]

    async def aembed_query(self, text: str) -> t.List[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: t.List[str]) -> t.List[t.List[float]]:
        return self.embed_documents(texts)


class Evaluator:
    def __init__(self):
        self.metrics = [faithfulness, answer_relevancy, context_recall]
        self.client = RagasLLMWrapper(LLMClient())
        self.embeddings = RagasEmbeddingWrapper()

    def evaluate(self, qa_pairs: list[dict]) -> list[dict]:
        data = {
            "question": [q["question"] for q in qa_pairs],
            "answer": [q["answer"] for q in qa_pairs],
            "contexts": [q["contexts"] for q in qa_pairs],
            "ground_truth": [q["ground_truth"] for q in qa_pairs],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.client,
            embeddings=self.embeddings,
        )
        return result.to_pandas().to_dict(orient="records")

    def save_report(self, results: list[dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
