## 🦙 LLaMA3-Summarizer

A high-quality summarization pipeline using Meta's **LLaMA 3** model (8B Instruct variant). This project demonstrates:
- Zero-shot summarization using prompting
- Supervised fine-tuning (SFT) for summarization
- LoRA-based parameter-efficient fine-tuning

---

## 🚀 Features

- ✨ Uses **Meta-LLaMA-3-8B-Instruct** from HuggingFace
- 📚 CNN/DailyMail summarization dataset
- 🧠 Zero-shot and fine-tuned summaries
- ⚙️ Config-driven training pipeline
- 🧪 ROUGE evaluation planned
- 🧵 LoRA support via `peft`

---

## 📦 Setup

```bash
git clone https://github.com/yourusername/LLaMA3-Summarizer.git
cd LLaMA3-Summarizer
pip install -r requirements.txt
```

---

## 📚 Dataset

This project uses the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail):

```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```

Each entry has:
- `article`: The news content
- `highlights`: The human-written summary

---

## 🧠 Zero-shot Summarization

```python
from model.summarize import load_model, summarize

tokenizer, model = load_model("meta-llama/Meta-Llama-3-8B-Instruct")
summary = summarize("Your long article text here...", tokenizer, model)
print(summary)
```

You can also run:

```bash
bash scripts/run_inference.sh
```

---

## 🏋️ Fine-Tuning (SFT)

To fine-tune the full model on CNN/DailyMail:

```bash
bash scripts/run_finetune.sh
```

Edit `model/finetune.py` to define your supervised training logic using HuggingFace `Trainer` or a custom loop.

---

## 🧪 LoRA Fine-Tuning

For memory-efficient fine-tuning using PEFT (LoRA):

```bash
bash scripts/run_finetune_lora.sh
```

LoRA reduces memory footprint by training adapter layers only. This is ideal for 8B models on limited GPUs.

---

## ⚙️ Config

Edit your training configuration in `configs/config.yaml`:

```yaml
model_name: meta-llama/Meta-Llama-3-8B-Instruct
max_input_length: 1024
max_output_length: 256
train_batch_size: 4
eval_batch_size: 4
num_train_epochs: 3
lr: 5e-5
lora: true
```

---

## 🔬 Roadmap

- [x] Zero-shot summarization
- [x] LoRA support
- [ ] Full training loop for fine-tuning
- [ ] ROUGE metric evaluation
- [ ] Streamlit/HF Space demo

---

## 📄 License

MIT License

---

## ✨ Acknowledgements

- Meta AI for LLaMA 3
- HuggingFace for Transformers & Datasets
- PEFT (LoRA) for efficient fine-tuning

---

## 🙌 Contributing

Feel free to open issues or pull requests. Let's build LLM tooling together!
