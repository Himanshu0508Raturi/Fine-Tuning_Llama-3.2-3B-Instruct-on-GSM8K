import os
import time
 
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

 
MODEL_ID = "raturihimanshu077/tinyllama-1.1b-gsm8k-nf4"
HF_TOKEN = os.environ.get("HF_TOKEN")
 
SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the word problem step by step, "
    "showing your reasoning clearly, then give the final numeric answer "
    "on its own line in the form: #### <answer>."
)
 
MAX_INPUT_TOKENS = 512      
MAX_NEW_TOKENS = 400

 
app = FastAPI(title="GSM8K Math Solver", version="1.0")
 
tokenizer = None
model = None
device = None
 
 
@app.on_event("startup")
def load_model():
    global tokenizer, model, device
 
    if HF_TOKEN is None:
        raise RuntimeError(
            "HF_TOKEN environment variable not set. This model repo is private — "
            "set HF_TOKEN before starting the server."
        )
 
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
 
    print(f"Model loaded on {device}.")
 
 
class SolveRequest(BaseModel):
    question: str = Field(..., min_length=1, description="A math word problem to solve.")
    max_new_tokens: int = Field(
        default=MAX_NEW_TOKENS, ge=16, le=1024,
        description="Max tokens to generate for the answer."
    )
 
 
class SolveResponse(BaseModel):
    question: str
    reasoning: str
    final_answer: str | None
    latency_seconds: float
  
import re
 
 
def extract_final_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\-0-9,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-0-9,\.]+", text)
    return numbers[-1].replace(",", "").strip() if numbers else None
 
 
 
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}
 
 
@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")
 
    token_count = len(tokenizer(req.question).input_ids)
    if token_count > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long ({token_count} tokens). Keep it under {MAX_INPUT_TOKENS} tokens.",
        )
 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.question},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
 
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start
 
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    )
 
    return SolveResponse(
        question=req.question,
        reasoning=generated,
        final_answer=extract_final_answer(generated),
        latency_seconds=round(latency, 2),
    )
 