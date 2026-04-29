from src.llm_client import LLMClient


class Generator:
    def __init__(self):
        self.client = LLMClient()

    def build_prompt(self, query: str, chunks: list[dict]) -> list[dict]:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"[文档片段 {i+1}]\n来源: {source}\n{chunk['text']}")
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """你是一个基于 Python 官方文档的问答助手。请根据提供的文档片段回答问题。
如果文档中没有相关信息，请明确说明"根据提供的文档，无法找到答案"。

输出格式要求：
1. 使用清晰的分点或分段结构，每个要点之间用空行分隔。
2. 适当使用 Markdown 粗体（如 **关键概念**）来突出核心术语。
3. 保持段落简洁，避免大段文字连续堆砌。
4. 所有来源引用统一放在答案末尾的"参考来源"部分，不要在正文中插入 [来源: xxx] 标记。"""

        user_prompt = f"""以下是与问题相关的文档片段：

{context}

问题: {query}

请基于以上文档片段，用清晰的分点/分段格式回答问题。"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate(self, query: str, chunks: list[dict], stream: bool = False) -> str:
        messages = self.build_prompt(query, chunks)
        if stream:
            return self.client.chat_stream(messages)
        response = self.client.chat(messages)
        return response.choices[0].message.content
