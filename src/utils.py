from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_gsm8k(split: str = "test"):
    ds = load_dataset("openai/gsm8k", "main")
    return ds[split]

def load_phi(model_name: str = "microsoft/phi-1_5"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # use float16 on GPU, float32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    return tokenizer, model

def generate_phi(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)
