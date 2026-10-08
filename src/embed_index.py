import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# BGE models work better with a prefix on the query side
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

class VectorIndex:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        emb = self.model.encode(texts, batch_size=32, show_progress_bar=True,
                                normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query, k=3):
        prefix = QUERY_PREFIX.get(self.model_name, "")
        q = self.model.encode([prefix + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            c = dict(self.chunks[idx])
            c["score"] = float(score)
            results.append(c)
        return results