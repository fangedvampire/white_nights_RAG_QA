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