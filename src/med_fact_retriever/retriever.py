from typing import Optional

from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever

from med_fact_retriever.data_model import Fact, RetrievedFacts


class MedFactRetriever:

    DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self, 
        retriever: VectorIndexRetriever, 
        use_reranker: bool = True, 
        top_n: int = 10, 
        use_fp16: bool = True,
    ) -> None:
        self.retriever = retriever
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = FlagEmbeddingReranker(
                model=self.DEFAULT_RERANKER, 
                top_n=top_n, 
                use_fp16=use_fp16,
            )
        else:
            self.reranker = None

    def __call__(self, query: str, rerank: bool = True) -> RetrievedFacts:
        nodes = self.retriever.retrieve(query)
        if self.use_reranker and rerank:
            nodes = self.reranker.postprocess_nodes(nodes, query_str=query)

        return RetrievedFacts(facts=[Fact.from_textnode(n) for n in nodes])
