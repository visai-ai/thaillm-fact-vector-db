from typing import List

from pydantic import BaseModel, Field

from llama_index.core.schema import TextNode


class Article(BaseModel):
    url: str = Field(..., description="Source article URL")
    title: str = Field(..., description="Article title")
    source_id: str = Field(..., description="Source article ID")


class Fact(BaseModel):
    fact_id: str = Field(..., description="Retrieved Fact ID")
    content: str = Field(..., description="Fact content")
    source_article: Article = Field(..., description="Source article that fact was extracted from")

    @classmethod
    def from_textnode(cls, node: TextNode) -> "Fact":
        text = node.text
        metadata = node.metadata

        return cls(
            fact_id=metadata["fact_id"],
            content=text,
            source_article=Article(
                url=metadata["article_url"],
                title=metadata["article_title"],
                source_id=metadata["source_id"]
            )
        )


class RetrievedFacts(BaseModel):
    facts: List[Fact] = Field(..., description="List of retrieved facts")


class InputQuery(BaseModel):
    query: str = Field(..., description="Input query")
    rerank: bool = Field(..., default_factory=lambda: True, description="State whether to rerank or not")
