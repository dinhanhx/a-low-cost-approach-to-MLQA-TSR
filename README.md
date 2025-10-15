# A low-cost low-energy approach to VQA on traffic signs problems

For VLSP 2025 MLQA-TSR, we propose a simple retrieval-based pipeline that requires no model training. Text and image features are extracted using Jina Embeddings v3, C-RADIOv2-B, and Owlv2, then stored in Qdrant for cosine similarity search. Retrieved examples directly provide legal terms for subtask 1 and serve as few-shot prompts for Llama 4 Maverick in subtask 2.

Our method achieved a top-5 ranking (F2 = 0.54) for retrieval and a top-1 ranking (accuracy = 0.86) for question answering.

## Setup

### Conda environment

```bash
sh scripts/conda_setup.sh
conda activate mlqa-tsr
sh scripts/pip_setup.sh
```

### Docker services

docker compose for qdrant
```yaml
name: qdrant
services:
    qdrant:
        ports:
            - 6333:6333
            - 6334:6334
        volumes:
            - ${PWD}/qdrant_storage:/qdrant/storage:z
        image: qdrant/qdrant
        restart: always
```

For vllm, it really depends on the GPU infrastructure.

See vllm and qdrant docs

## Run

At `notebooks` folder,

Read `detect_image_object.ipynb`, `extract_image_feature.ipynb` firstly, then run it

Read `index_qdrant.ipynb` secondly, then run it

Read `naive_vector_search.ipynb` for subtask 1, then run it

Read `vlm_answer.ipynb` for subtask 2, then run it

The `src` folder contains reusable code such as processing data, inferring models, connecting databases.

## Citation

```
Waiting for INLG 2025 public paper link
```