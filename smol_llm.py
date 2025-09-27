import re
import reasoning_gym
from rich import print
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

system_prompt = (
    "Generate an answer after thinking.\n"
    "Use <think> your reasoning process </think> <answer> your answer </answer> tags to structure your response.\n"
    "You must answer only within <answer>...</answer> tags."
)


def extract_answer(response: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else None


def generate_response(question: str, system: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    chat_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,  # return a formatted string, not token IDs
    )
    print(f"[bold green]Chat Template:[/bold green] {chat_str}")

    inputs = tokenizer(chat_str, return_tensors="pt")
    print(f"[bold green]Inputs:[/bold green] {inputs['input_ids'][:10]}")

    outputs = llm.generate(**inputs, max_new_tokens=100)
    input_length = inputs["input_ids"].shape[1]
    newly_generated_tokens = outputs[0, input_length:]

    decoded = tokenizer.batch_decode(newly_generated_tokens)[0]
    return decoded


if __name__ == "__main__":
    environment_name = "propositional_logic"
    dataset = reasoning_gym.create_dataset(environment_name, size=1, seed=100)

    for entry in dataset:
        question = entry["question"]
        print(f"[bold blue]Question:[/bold blue] {question}")

        response = generate_response(question, system_prompt)
        print(f"[bold white]Model Output:[/bold white] {response}")

        model_answer = extract_answer(response)
        print(f"[bold white]Extracted Answer:[/bold white] {model_answer}")

        reward = dataset.score_answer(answer=model_answer, entry=entry)
        if reward > 0:
            print(f"[bold green]Reward:[/bold green] {reward}")
        else:
            print(f"[bold red]Reward:[/bold red] {reward}")
