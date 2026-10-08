import os
import re
import fitz  # PyMuPDF

START_PAT = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
END_PAT = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", re.I)

def load_pdfs(folder):
    pages = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".pdf"):
            continue

        doc = fitz.open(os.path.join(folder, fname))

        """# 1. extract and clean whitespace, keep original page numbers
        raw = []
        for i, page in enumerate(doc):
            text = " ".join(page.get_text().split())
            if text:
                raw.append([i + 1, text])

        # 2. cut everything before the START marker (if present)
        start_idx = 0
        for idx, (_, text) in enumerate(raw):
            m = START_PAT.search(text)
            if m:
                raw[idx][1] = text[m.end():].strip()
                start_idx = idx
                break

        # 3. cut everything after the END marker (if present)
        end_idx = len(raw)
        for idx in range(start_idx, len(raw)):
            m = END_PAT.search(raw[idx][1])
            if m:
                raw[idx][1] = raw[idx][1][:m.start()].strip()
                end_idx = idx + 1
                break

        # 4. keep the pages in range
        for page_no, text in raw[start_idx:end_idx]:
            if text:
                pages.append({"source": fname, "page": page_no, "text": text})
    return pages"""
        for i in range(4, min(56, len(doc))):
            text = " ".join(doc[i].get_text().split())

            if text:
                pages.append({
                    "source": fname,
                    "page": i + 1,
                    "text": text
                })

    return pages

if __name__ == "__main__":
    pages = load_pdfs("data/pdfs")
    print(f"Loaded {len(pages)} pages")
    print("First page:", pages[0]["page"], "-", pages[0]["text"][:300])
    print("Last page:", pages[-1]["page"], "-", pages[-1]["text"][-300:])

