import re
from src.ingest import load_pdfs
from src.chunking import chunk_pages
from src.embed_index import VectorIndex

pages = load_pdfs("data/pdfs")
full_text = " ".join(p["text"] for p in pages)

# 1. Does the word exist in the book, and how often?
for word in ["Petersburg", "Peter", "city", "St\\. Petersburg", "grandmother", "pinned"]:
    hits = [m.start() for m in re.finditer(rf"\b{word}\b", full_text, re.I)]
    print(f"{word!r}: {len(hits)} occurrences")
    if hits:
        s = hits[0]
        print("   e.g.:", full_text[max(0, s-80): s+120])

# 2. Retrieval check with whole-word matching
chunks = chunk_pages(pages, chunk_size=100, overlap=20)
index = VectorIndex("sentence-transformers/all-MiniLM-L6-v2")
index.build(chunks)

tests = [
    ("Who is Nastenka?", ["Nastenka"]),
    ("What city does the story take place in?", ["Petersburg"]),
    ("Why does the grandmother pin Nastenka's dress?", ["pinned", "pin"]),
]

for q, kws in tests:
    res = index.search(q, k=5)
    text = " ".join(r["text"] for r in res)
    found = [k for k in kws if re.search(rf"\b{k}\b", text, re.I)]
    print(f"\nQ: {q}\n   whole-word keywords in top 5: {found}")