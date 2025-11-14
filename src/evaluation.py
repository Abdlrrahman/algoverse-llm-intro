import re
from tqdm.auto import tqdm

from .utils import generate_phi
from .prompts import prompt_direct, prompt_cot, prompt_custom_checker

def extract_final_answer(text: str):
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None

def eval_prompt_on_dataset(dataset, model, tokenizer, prompt_fn, n_examples=50, do_sample=False):
    correct = 0
    total = 0
    for ex in tqdm(dataset.select(range(n_examples))):
        q = ex["question"]
        gold = extract_final_answer(ex["answer"])
        prompt = prompt_fn(q)
        output = generate_phi(model, tokenizer, prompt, do_sample=do_sample)
        pred = extract_final_answer(output)

        if gold is not None and pred == gold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0
