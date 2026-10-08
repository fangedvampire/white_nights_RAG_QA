import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Generator:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        self.model.eval()

    def build_prompt(self, question, contexts):
        context = "\n\n".join(
            f"[{i+1}] (page {c['page']}) {c['text']}" for i, c in enumerate(contexts)
        )
        messages = [
            {"role": "system",
             "content": "You answer questions about a novel using only the excerpts provided. "
                        "Answer in one or two short sentences. "
                        "If the excerpts do not contain the answer, say you could not find it."},
            {"role": "user",
             "content": f"Excerpts:\n{context}\n\nQuestion: {question}"},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def answer(self, question, contexts, max_new_tokens=100):
        prompt = self.build_prompt(question, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()