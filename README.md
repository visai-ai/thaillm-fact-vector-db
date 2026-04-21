# ThaiLLM - Medical Facts Retriever Tools
Retrieval endpoint for medical fact retrieval tools. This endpoint will be integrated with MCP servers for Thai-LLM medical tool.

## Usage
We wrapped everything into a docker, so you can basically run these commands:

1. Build docker image
```sh
docker build -t thaillm/med-fact-server .
```

2. Start server
```sh
# change gpu device here
docker run -it --gpus '"device=3"' -p 41000:41000 --shm-size=4G -v $PWD/cache:/app/cache --name med-fact-server thaillm/med-fact-server
```

## Requirements
We suggest using GPU-based server for faster rerank and faster vector ingestation. Here're the hard requirements
- Python 3.13+
- Nvidia GPU (Requires around 11-12GB of memory)

For our dev environment,
- NVIDIA A100 40GB

## API Endpoints Usage

### python

```python
import requests
import json

url = "http://localhost:41000/query"

payload = json.dumps({
  "query": "โซเดียมเยอะเกินส่งผลให้เกิดอะไรได้บ้าง",
  "rerank": True
})
headers = {
  'Authorization': 'Bearer <api-key>',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

### curl

```sh
curl --location 'http://10.204.100.71:41000/query' \
--header 'Authorization: Bearer $API_KEY' \
--header 'Content-Type: application/json' \
--data '{
    "query": "โซเดียมเยอะเกินส่งผลให้เกิดอะไรได้บ้าง",
    "rerank": true
}'
```

## Environment Variables
- `HF_TOKEN`: Can leave empty once the fact repositories are public. This is set only during development.
- `TOP_K`: Integer, maximum number of retrieved top_k. Default at 30.
- `USE_RERANKER`: Boolean, specify whether to load reranker or not. This saves a bit of GPU memory. Default as True.
- `RERANK_TOP_N`: Top N documents to be selected after rerank. Note that `RERANK_TOP_N` must less than `TOP_K`. Default as 10.
- `NPROBES`: nprobes config for LanceBD. The higher the value, the more accurate the result, but with slower retrieval time. Default at 20.
- `CACHE_DIR`: Default directory to store cache. Typically the cache size is around 1GB. Default at container path `/app/cache`.
