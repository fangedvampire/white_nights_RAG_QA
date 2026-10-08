import argparse
from src.pipeline import RAGPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--pdfs", default="data/pdfs")
parser.add_argument("--embed", default="sentence-transformers/all-MiniLM-L6-v2")
parser.add_argument("--chunk", type=int, default=150)
args = parser.parse_args()

rag = RAGPipeline(args.pdfs, args.embed, args.chunk)

while True:
    q = input("\nAsk a question (or 'quit'): ").strip()
    if q.lower() in {"quit", "exit"}:
        break
    res = rag.ask(q)
    print("\nAnswer:", res["answer"])
    print("Sources:")
    for s in res["sources"]:
        print(f"  - {s['source']} (page {s['page']}, score {s['score']})")