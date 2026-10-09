import json
import os

def chunk_pages(pages, chunk_size=200, overlap=40):
    chunks = []
    step = chunk_size - overlap
    for p in pages:
        words = p["text"].split()
        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]
            chunks.append({
                "text": " ".join(piece),
                "source": p["source"],
                "page": p["page"],
            })
            if start + chunk_size >= len(words):
                break
    return chunks


def save_chunks(chunks, chunk_size, overlap, out_dir="data/chunks"):
    os.makedirs(out_dir, exist_ok=True)

    # 1. JSON: structured, reloadable
    json_path = os.path.join(out_dir, f"chunks_{chunk_size}w_{overlap}o.json")
    numbered = [{"chunk_id": i, **c} for i, c in enumerate(chunks)]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(numbered, f, ensure_ascii=False, indent=2)

    # 2. TXT: easy to read and scroll through
    txt_path = os.path.join(out_dir, f"chunks_{chunk_size}w_{overlap}o.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for c in numbered:
            f.write(f"=== chunk {c['chunk_id']} | page {c['page']} | "
                    f"{len(c['text'].split())} words ===\n{c['text']}\n\n")

    return json_path, txt_path


def load_chunks(chunk_size, overlap, in_dir="data/chunks"):
    path = os.path.join(in_dir, f"chunks_{chunk_size}w_{overlap}o.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from src.ingest import load_pdfs

    pages = load_pdfs("data/pdfs")
    for size, overlap in [(100, 20), (150, 30), (300, 60)]:
        chunks = chunk_pages(pages, chunk_size=size, overlap=overlap)
        json_path, txt_path = save_chunks(chunks, size, overlap)
        print(f"chunk_size={size}, overlap={overlap}: {len(chunks)} chunks -> {txt_path}")