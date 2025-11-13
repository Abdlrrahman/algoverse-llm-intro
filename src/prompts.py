def prompt_direct(question: str) -> str:
    return f"Question: {question}\nAnswer:"

def prompt_cot(question: str) -> str:
    return (
        "You are a careful math tutor.\n"
        f"Question: {question}\n"
        "Let's think step by step and show all intermediate calculations.\n"
        "At the end, write the final answer as: Answer: <number>\n"
        "Reasoning:"
    )

def prompt_custom_checker(question: str) -> str:
    return (
        "You are an expert math teacher.\n"
        f"Question: {question}\n"
        "First, solve the problem step by step.\n"
        "Then, review your steps and correct any mistakes.\n"
        "Finally, give the final answer in the format: Answer: <number>\n"
        "Reasoning and solution:"
    )
