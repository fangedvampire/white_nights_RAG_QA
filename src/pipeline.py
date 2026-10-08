from src.ingest import load_pdfs
from src.chunking import chunk_pages
from src.embed_index import VectorIndex
from src.generate import Generator

class RAGPipeline:
    def __init__(self, pdf_folder, embed_model, chunk_size, overlap=40,
                 generator=None, top_k=3):
        self.top_k = top_k
        pages = load_pdfs(pdf_folder)
        chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        self.index = VectorIndex(embed_model)
        self.index.build(chunks)
        self.generator = generator or Generator()

    def ask(self, question):
        contexts = self.index.search(question, k=self.top_k)
        answer = self.generator.answer(question, contexts)
        sources = [
            {"source": c["source"], "page": c["page"], "score": round(c["score"], 3)}
            for c in contexts
        ]
        return {"question": question, "answer": answer,
                "sources": sources, "contexts": contexts}