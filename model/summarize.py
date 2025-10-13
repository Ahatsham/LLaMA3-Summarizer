from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return tokenizer, model

def summarize(text, tokenizer, model, max_new_tokens=256):
    prompt = f"[INST] Summarize the following article based on the content:\n\n{text} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    summary_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
