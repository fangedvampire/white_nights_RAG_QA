import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_name="google/flan-t5-base", max_input_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.max_input_tokens = max_input_tokens

    def build_prompt(self, question, contexts):
        header = "Read the context and answer the question.\n\nContext:\n"
        footer = f"\n\nQuestion: {question}\nAnswer:"

        fixed = len(self.tokenizer(header + footer, add_special_tokens=False).input_ids)
        budget = self.max_input_tokens - fixed - 2
        per_chunk = max(budget // max(len(contexts), 1), 20)

        trimmed = []
        for c in contexts:
            ids = self.tokenizer(c["text"], add_special_tokens=False).input_ids[:per_chunk]
            trimmed.append(self.tokenizer.decode(ids))

        return header + "\n\n".join(trimmed) + footer

    def answer(self, question, contexts, max_new_tokens=128):
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                truncation=True, max_length=self.max_input_tokens)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)