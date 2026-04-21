import os
from dataclasses import dataclass


@dataclass
class AppSettings:

    top_k: int = 30
    use_reranker: bool = True # set to false of you want to spare some memory
    rerank_top_n: int = 10
    nprobes: int = 20
    cache_dir: str = "/app/cache"
    include_golden_fact: bool = False

    @classmethod
    def from_env(cls) -> "AppSettings":
        inc_g = os.getenv("INCLUDE_GOLDEN_FACT", None)
        top_k = int(os.getenv("TOP_K", 32))
        use_reranker = bool(os.getenv("USE_RERANKER", True))
        rerank_top_n = int(os.getenv("RERANK_TOP_N", 32))
        nprobes = int(os.getenv("NPROBES", 20))
        cache_dir = os.getenv("CACHE_DIR", "/app/cache")
        include_golden_fact = True if isinstance(inc_g, str) and inc_g.lower() == "true" else False

        return cls(
            top_k=top_k,
            use_reranker=use_reranker,
            rerank_top_n=rerank_top_n,
            nprobes=nprobes,
            cache_dir=cache_dir,
            include_golden_fact=include_golden_fact
        )
