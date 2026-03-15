# Embeddings Pipeline

Generates vector embeddings for real estate listings and stores them in a FAISS index for similarity search.

## Requirements

- [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled
- `faiss-cpu`, `httpx`, `tqdm`, `numpy`

```bash
ollama pull nomic-embed-text
uv add faiss-cpu httpx tqdm numpy
```

## How it works

Each listing is converted into a textual representation:

```python
f"Title: {row['title']}, Price: {row['price']} {row['currency']}, 
Description_in_native: {row['description_native']}, 
Description_in_english: {row['description_en']}"
```

Embeddings are generated via Ollama's API using `nomic-embed-text` (768-dim), with async requests and a semaphore to control concurrency. Failed embeddings (e.g. text too long) fall back to zero vectors.

The resulting FAISS `IndexFlatL2` index is saved to disk:

```bash
faiss.write_index(index, 'index')
```

## Usage

```python
# start ollama first
# ollama serve

embeddings = await build_embeddings(df['textual_representation'].tolist())
X = np.array(embeddings, dtype='float32')
index.add(X)
faiss.write_index(index, 'index')
```

To reload:
```python
index = faiss.read_index('index')
```