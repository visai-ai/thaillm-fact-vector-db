import os
from typing import List

import torch.cuda
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore



class MedFactVectorIndexer:

    DEFAULT_TABLE_NAME = "medfact_bgem3_dense"
    DEFAULT_MODEL_NAME = "BAAI/bge-m3"
    DEFAULT_DATA_DIR = "lancedb_data"
    DEFAULT_PERSIST_DIR = "li_store"

    def __init__(
        self,
        cache_dir: str,
        table_name: str = DEFAULT_TABLE_NAME,
        model_name: str = DEFAULT_MODEL_NAME,
        nprobes: int = 20
    ) -> None:
        self.table_name = table_name
        self.cache_dir = cache_dir
        self.nprobes = nprobes
        self.model_name = model_name
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.uri, exist_ok=True)

        device = "cuda" if self.is_cuda_available() else "cpu"
        Settings.embed_model = HuggingFaceEmbedding(model_name, device=device, cache_folder=self.cache_dir)
        self.vstore = LanceDBVectorStore(uri=self.uri, table_name=self.table_name, nprobes=self.nprobes)

        if self.persistent_created():
            self.load_from_storage()
        else:
            print("Persistent storage hasn't been loaded")

    @property
    def uri(self) -> str:
        return os.path.join(self.cache_dir, self.DEFAULT_DATA_DIR)

    @property
    def persist_dir(self) -> str:
        return os.path.join(self.cache_dir, self.DEFAULT_PERSIST_DIR)

    def is_cuda_available(self) -> bool:
        return torch.cuda.device_count() > 0

    def is_indexed(self) -> bool:
        return hasattr(self, "storage") and hasattr(self, "index")

    def persistent_created(self) -> bool:
        return os.path.exists(self.persist_dir)

    def ingest(self, nodes: List[TextNode], show_progress: bool = True) -> None:
        self.storage = StorageContext.from_defaults(vector_store=self.vstore)
        self.index = VectorStoreIndex(nodes, storage_context=self.storage, show_progress=show_progress)
        self.index.storage_context.persist(persist_dir=self.persist_dir)

    def load_from_storage(self) -> None:
        self.storage = StorageContext.from_defaults(vector_store=self.vstore, persist_dir=self.persist_dir)
        self.index = load_index_from_storage(self.storage, persist_dir=self.persist_dir)

    def get_retriever(self, top_k: int = 30) -> VectorIndexRetriever:
        return self.index.as_retriever(similarity_top_k=top_k)
