import json
import itertools
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from src.pipeline import RAGPipeline
from src.generate import Generator

PDF_DIR = "data/pdfs"
CHUNK_SIZES = [100, 200]
EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

def keyword_fraction(text, keywords):
    text = text.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)

def evaluate_config(rag, test_set, judge):
    hits, correct, relevance = [], [], []
    for item in test_set:
        res = rag.ask(item["question"])
        retrieved_text = " ".join(c["text"] for c in res["contexts"])

        hits.append(keyword_fraction(retrieved_text, item["keywords"]) >= 0.5)
        correct.append(keyword_fraction(res["answer"], item["keywords"]) >= 0.5)

        emb = judge.encode([res["answer"], item["reference"]], convert_to_tensor=True)
        relevance.append(float(util.cos_sim(emb[0], emb[1])))

    n = len(test_set)
    return {
        "retrieval_hit_%": round(100 * sum(hits) / n, 1),
        "correct_%": round(100 * sum(correct) / n, 1),
        "avg_relevance": round(sum(relevance) / n, 3),
    }

if __name__ == "__main__":
    with open("eval/test_questions.json") as f:
        test_set = json.load(f)

    generator = Generator()  # load once, reuse
    judge = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    rows = []
    for chunk_size, model in itertools.product(CHUNK_SIZES, EMBED_MODELS):
        print(f"\n=== chunk={chunk_size} | model={model} ===")
        rag = RAGPipeline(PDF_DIR, model, chunk_size, generator=generator)
        metrics = evaluate_config(rag, test_set, judge)
        rows.append({"chunk_size": chunk_size, "embed_model": model, **metrics})
        print(metrics)

    df = pd.DataFrame(rows)
    df.to_csv("results/results.csv", index=False)
    print("\n", df.to_string(index=False))