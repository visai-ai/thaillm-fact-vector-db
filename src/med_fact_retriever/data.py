from typing import List, Dict, Union, Literal, Optional

from datasets import load_dataset

from llama_index.core.schema import TextNode

# custom schema
_Subfact = Dict[Literal["text", "supporting_lines"], Union[str, List[str]]]
_Validation = Dict[Literal["grounded", "subfacts"], Union[bool, List[_Subfact]]]
_Fact = Dict[Literal["fact_id", "text", "validation", "source_id"], Union[str, _Validation]]
_ArticleSource = Dict[Literal["title", "url"], Optional[str]]
_FactWithSource = Dict[Literal["fact_id", "text", "validation", "source_id", "source"], Union[str, _Validation, _ArticleSource]]

BGEM3_NAME = "BAAI/bge-m3"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"


class MedFactDatabase:

    DATASET_NAME = "ThaiLLM/med-facts"
    ARTICLE_NAME = "ThaiLLM/med-articles"

    GOLDEN_ARTICLES = "ThaiLLM/med-qas-golden-articles"
    GOLDEN_QAS = "ThaiLLM/med-qas-golden"

    def __init__(self, include_golden_fact: bool = False, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = cache_dir
        self.include_golden_fact = include_golden_fact
        self._setup()

    def _setup_golden(self) -> None:
        self.golden_facts = None

        if self.include_golden_fact:
            golden_qas = load_dataset(self.GOLDEN_QAS)
            golden_articles = load_dataset(self.GOLDEN_ARTICLES)
            
            # populate golden facts
            self.golden_facts = []

            if len(golden_qas) == 1:
                article_mapper = {a["aid"]: a for a in golden_articles["train"]}
                for qa in golden_qas["train"]:
                    for fact in qa["facts"]:
                        article = article_mapper[fact["aid"]]
                        self.golden_facts.append({
                            "fact_id": fact["fid"],
                            "text": fact["content"],
                            "source": {
                                "title": article["title"],
                                "url": article["url"]
                            },
                            "source_id": article["source_id"]
                        })
            else:
                assert "test" in golden_qas
                assert "val" in golden_qas
                for k in ["val", "test"]:
                    article_mapper = {a["aid"]: a for a in golden_articles["train"]}
                    for qa in golden_qas[k]:
                        for fact in qa["facts"]:
                            article = article_mapper[fact["aid"]]
                            self.golden_facts.append({
                                "fact_id": fact["fid"],
                                "text": fact["content"],
                                "source": {
                                    "title": article["title"],
                                    "url": article["url"]
                                },
                                "source_id": article["source_id"]
                            })

    def _setup(self) -> None:
        self.facts = load_dataset(self.DATASET_NAME, cache_dir=self.cache_dir)
        self.facts = self.facts["train"]

        self.articles = load_dataset(self.ARTICLE_NAME, cache_dir=self.cache_dir)
        self.articles = self.articles["train"]
        self.articles = {a["aid"]: a for a in self.articles}

        self.facts = self.facts.map(self.resolve_fact_source)

        self._setup_golden()

    def get_aid_from_fid(self, fid: str) -> str:
        aid = fid.split("_")[1].split("-")[0]
        if aid not in self.articles:
            raise ValueError(f"Cannot find article id {aid}")
        return aid

    def resolve_fact_source(self, example: _Fact) ->_FactWithSource:
        aid = self.get_aid_from_fid(example["fact_id"])

        article = self.articles.get(aid)
        if article is None:
            raise ValueError(f"Cannot find article id {aid}")

        metadata = article["metadata"]
        title = metadata.get("title", None)
        url = metadata.get("url", None)

        example["source"] = {
            "title": title,
            "url": url
        }
        return example

    def __len__(self) -> int:
        return len(self.facts)

    def export_text_node(self) -> List[TextNode]:
        text_nodes = [
            TextNode(
                text=fact["text"],
                metadata={
                    "fact_id": fact["fact_id"],
                    "article_title": fact["source"]["title"],
                    "article_url": fact["source"]["url"],
                    "source_id": fact["source_id"]
                }
            )
            for fact in self.facts
        ]

        if self.include_golden_fact:
            text_nodes.extend([
                    TextNode(
                    text=fact["text"],
                    metadata={
                        "fact_id": fact["fact_id"],
                        "article_title": fact["source"]["title"],
                        "article_url": fact["source"]["url"],
                        "source_id": fact["source_id"]
                    }
                )
                for fact in self.golden_facts
            ])

        return text_nodes
