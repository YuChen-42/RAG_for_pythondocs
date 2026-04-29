import os
from openai import OpenAI
from src import config


class QueryRewriter:
    def __init__(self):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY or os.getenv("LLM_API_KEY", "")
        )
        self.model = config.LLM_MODEL

    def split_multi_topic(self, query: str) -> list[str]:
        prompt = f"""判断以下问题是否涉及多个完全不同的主题/模块。
如果是，请拆分为多个独立子问题，每行一个。
如果不是，请原样返回问题本身。

问题: {query}

只输出问题列表，不要任何解释。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        return lines if lines else [query]

    def expand_query(self, query: str) -> list[str]:
        prompt = f"""将以下问题改写为 2-3 个语义等价但表述不同的查询，每行一个。
原问题: {query}
只输出查询列表，不要任何解释。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
        return lines if lines else [query]

    def rewrite(self, query: str, use_query_rewrite: bool = None) -> list[str]:
        if use_query_rewrite is None:
            use_query_rewrite = config.USE_QUERY_REWRITE
        if not use_query_rewrite:
            return [query]
        if config.USE_MULTI_QUERY_SPLIT:
            sub_queries = self.split_multi_topic(query)
        else:
            sub_queries = [query]
        all_queries = []
        for sq in sub_queries:
            all_queries.extend(self.expand_query(sq))
        return list(dict.fromkeys(all_queries))
