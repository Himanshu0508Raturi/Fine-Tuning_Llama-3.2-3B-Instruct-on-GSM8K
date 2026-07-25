---
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
library_name: peft
license: apache-2.0
datasets:
  - openai/gsm8k
language:
  - en
pipeline_tag: text-generation
tags:
  - lora
  - qlora
  - peft
  - math-reasoning
  - gsm8k
  - tinyllama
---

# TinyLlama-1.1B-Chat Fine-Tuned on GSM8K (QLoRA / NF4)

## Model Details

### Model Description

This model is a fine-tuned version of `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, adapted to
solve grade-school math word problems with explicit step-by-step reasoning. It was
fine-tuned using QLoRA (4-bit NF4 quantization + LoRA adapters) on the GSM8K training
set, then evaluated against the base model on the GSM8K test set to measure the accuracy
improvement from fine-tuning.

The model is trained to respond to a math word problem with a full chain-of-thought
explanation, ending in a final numeric answer on its own line in the form `#### <answer>`.

- **Developed by:** Himanshu Raturi
- **Model type:** Causal decoder-only language model, fine-tuned with LoRA adapters (PEFT)
- **Language(s):** English
- **License:** Apache 2.0 (inherited from TinyLlama base model)
- **Finetuned from model:** [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

### Model Sources

- **Repository:** https://huggingface.co/raturihimanshu077/tinyllama-1.1b-gsm8k-nf4
- **Base model paper:** [TinyLlama: An Open-Source Small Language Model](https://arxiv.org/abs/2401.02385)
- **Dataset paper:** [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (GSM8K)

## Uses

### Direct Use

Solving grade-school-level math word problems with step-by-step reasoning. Intended for
educational tools, tutoring assistants, and as a lightweight reference implementation of
QLoRA fine-tuning for reasoning tasks.

### Downstream Use

Can be merged into applications that need a small, low-cost math-reasoning component —
e.g. as a backend model behind a tutoring API (see the accompanying FastAPI service in
the project repository).

### Out-of-Scope Use

- Not intended for open-domain question answering, general chat, or tasks unrelated to
  math word problems — the fine-tuning data is narrow and will not generalize well outside it.
- Not suitable for high-stakes numeric decisions (financial, medical, engineering
  calculations) — this is a small model fine-tuned on grade-school problems and should
  not be relied on for correctness in real-world quantitative tasks without verification.
- Not evaluated for problems requiring multi-step algebra, calculus, or symbolic math
  beyond GSM8K's scope.

## Bias, Risks, and Limitations

- Inherits any biases present in the TinyLlama base model and its pretraining data.
- As a 1.1B-parameter model, it has materially lower reasoning capacity than larger
  models — expect more frequent arithmetic and logical errors, especially on problems
  with more steps or unusual phrasing than typical GSM8K examples.
- Fine-tuned only on GSM8K-style problems (grade-school arithmetic word problems);
  performance on other math domains (e.g. algebra, geometry) is untested and likely poor.
- Like all LLMs, may produce confident-sounding but incorrect reasoning ("hallucinated"
  steps) — outputs should be checked, not trusted blindly, particularly for anything
  beyond casual/educational use.

### Recommendations

Users should verify the model's final numeric answers independently for any use case
where correctness matters, and should not assume the step-by-step reasoning is always
logically sound just because it reads fluently.

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "raturihimanshu077/tinyllama-1.1b-gsm8k-nf4"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the word problem step by step, "
    "showing your reasoning clearly, then give the final numeric answer "
    "on its own line in the form: #### <answer>."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"},
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
outputs = model.generate(input_ids=inputs, max_new_tokens=400, do_sample=False)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Training Details

### Training Data

[GSM8K](https://huggingface.co/datasets/openai/gsm8k) ("main" config), official split:
7,473 training examples, 1,319 test examples. Each example is a grade-school math word
problem paired with a human-written step-by-step solution ending in a final numeric answer.

### Training Procedure

#### Preprocessing

- GSM8K's `<<calculator annotation>>` markup was stripped from the reasoning text to
  produce clean natural-language solutions.
- Each example was formatted into TinyLlama's native chat template (system + user +
  assistant turns), with the assistant turn containing the cleaned step-by-step solution.

#### Training Hyperparameters

- **Training regime:** 4-bit NF4 quantized base weights (bitsandbytes `BitsAndBytesConfig`,
  double quantization enabled), bf16 compute dtype, LoRA adapters trained in mixed precision
- **LoRA config:** r=16, alpha=16, dropout=0.05, target modules: `q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Epochs:** 3
- **Effective batch size:** 16 (`per_device_train_batch_size=8` × `gradient_accumulation_steps=2`)
- **Learning rate:** 2e-4, linear schedule
- **Optimizer:** AdamW (8-bit paged variant)

#### Speeds, Sizes, Times

- Trained on a single Colab GPU
- LoRA adapter size: a few MB (only ~0.1–1% of total parameters trainable)
- Merged model size: ~2.2GB (bf16)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

GSM8K official test split (1,319 examples), same split used for both baseline and
fine-tuned evaluation to ensure a fair comparison.

#### Metrics

Exact-match accuracy on the final numeric answer, extracted via the `#### <answer>`
marker from both the model's output and GSM8K's gold solution.

### Results

| Model | Exact-match accuracy (GSM8K test) |
|---|---|
| Base TinyLlama-1.1B-Chat (zero-shot, CoT-prompted) | *TODO: fill in your number* |
| Fine-tuned (this model) | *TODO: fill in your number* |

#### Summary

*TODO: one or two sentences once you have the final numbers — e.g. "Fine-tuning improved
exact-match accuracy from X% to Y% on the GSM8K test set, demonstrating that small models
benefit substantially from task-specific fine-tuning even when prompted with explicit
chain-of-thought instructions."*

## Technical Specifications

### Model Architecture and Objective

Decoder-only transformer (Llama architecture), causal language modeling objective,
fine-tuned with LoRA low-rank adapters on top of 4-bit quantized base weights.

### Compute Infrastructure

#### Hardware

Single GPU (Google Colab)

#### Software

- `transformers`, `peft`, `bitsandbytes`, `trl` (TRL `SFTTrainer`)
- PEFT 0.13.2

## Citation

If you use this model, please cite the original GSM8K and TinyLlama papers:

**BibTeX:**
```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}

@article{zhang2024tinyllama,
  title={TinyLlama: An Open-Source Small Language Model},
  author={Zhang, Peiyuan and Zeng, Guangtao and Wang, Tianduo and Lu, Wei},
  journal={arXiv preprint arXiv:2401.02385},
  year={2024}
}
```

## Model Card Contact

Himanshu Raturi — [Hugging Face profile](https://huggingface.co/raturihimanshu077)

### Framework versions

- PEFT 0.13.2
