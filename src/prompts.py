def prompt_direct(question: str) -> str:
    return f"Question: {question}\nAnswer:"

def prompt_cot(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Let's think step by step and show intermediate reasoning.\n"
        "At the end write final answer as: Answer: <number>\n"
        "Reasoning:"
    )

def prompt_custom_checker(question: str) -> str:
    return (
        f"Question: {question}\n"
        "First, solve step-by-step.\n"
        "Then review your reasoning and fix mistakes.\n"
        "Finally give the answer as: Answer: <number>\n"
        "Reasoning:"
    )
