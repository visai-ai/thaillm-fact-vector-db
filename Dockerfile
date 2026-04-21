FROM python:3.13-slim

WORKDIR /root
RUN apt-get update && apt-get upgrade -y && apt-get install -y git
RUN git clone https://github.com/run-llama/llama_index.git

RUN apt-get install gcc cmake build-essential -y
RUN pip install sentencepiece

# install llamaindex indexer, reranker, and lancedb compatibility
RUN pip install -e /root/llama_index/llama-index-core  
RUN pip install -e /root/llama_index/llama-index-integrations/embeddings/llama-index-embeddings-huggingface
RUN pip install -e /root/llama_index/llama-index-integrations/postprocessor/llama-index-postprocessor-flag-embedding-reranker
RUN pip install -e /root/llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-lancedb

# install directories
WORKDIR /app
COPY . .
RUN pip install -e .

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "41000"]