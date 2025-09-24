import reasoning_gym
import openai
import re 
from rich import print
from dotenv import load_dotenv

load_dotenv()

SEED = 100

"""Example usage of the Leg Counting dataset.
data = reasoning_gym.create_dataset('leg_counting', size=1, seed=42)
for i, x in enumerate(data):
    print(f'{i}: q="{x['question']}", a="{x['answer']}"')
    print('metadata:', x['metadata'])
    # use the dataset's `score_answer` method for algorithmic verification
    assert data.score_answer(answer=x['answer'], entry=x) == 1.0
"""


# Create out dataset/ the environment for our LLM to interact with
environment_name =  "propositional_logic"
dataset = reasoning_gym.create_dataset(environment_name, size=5, seed=SEED)

# System prompt for the LLM

system_prompt = """
Generate an answer after thinking. 
Use <think> your reasoning process </think> <answer> your answer </answer> tags to structure your response.
You must answer only within <answer>... </answer> tags.
"""

# LLM is then able to output its reasoning process then answer.

# Extract the answer from the LLM's response

def extract_answer(response):
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    return match.group(1).strip() if match else None



# loop through the dataset
client = openai.OpenAI()

""" Check if llm is working
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
"""

for example in dataset:
    question = example['question']
    answer = example['metadata']['example_answer']
    metadata = example['metadata']

    print(f"[bold white]System Prompt:[/bold white] {system_prompt}")
    print(f"[bold blue]Question:[/bold blue] {question}")
    print(f"[bold green]True Answer:[/bold green] {answer}")
    

    # query the llm
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    # print out its thinking process
    # print out response
    # print out just the answer using the extract_answer function
    print(f"[bold white]Thinking Process:[/bold white] {response.choices[0].message.content}")
    print(f"[bold white]Answer:[/bold white] {extract_answer(response.choices[0].message.content)}")

    # Extract the answer from the LLM's response
    extracted_answer = extract_answer(response.choices[0].message.content)
    print(f"[bold white]Extracted Answer:[/bold white] {extracted_answer}")

    score_func = reasoning_gym.get_score_answer_fn(example["metadata"]["source_dataset"])

    reward = score_func(extracted_answer, example)
    if reward > 0:
        print(f"[bold green]Reward:[/bold green] {reward}")
    else:
        print(f"[bold red]Reward:[/bold red] {reward}")