import re
import torch
import reasoning_gym
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from rich import print
from dotenv import load_dotenv
import wandb # for logging
from datasets import load_dataset
from datasets import Dataset
import re

wandb.login()

# use a dataset from reasoning_gym
environment_name = "propositional_logic"
dataset = reasoning_gym.create_dataset(environment_name, size=5, seed=100)
train_dataset = Dataset.from_list([
    {"question": x["question"], "entry": x} for x in dataset
]) # train_dataset is a dataset of questions and their corresponding entry, 
   # the entry is the full reasoning_gym dataset entry

# 2) Prompt format with <think>/<answer>
SYSTEM_PROMPT = (
    "Generate an answer after thinking.\n"
    "Use <think> your reasoning process </think> <answer> your answer </answer> tags.\n"
    "You must output only within <answer>...</answer>."
)

def extract_answer(response: str) -> str | None:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else None

# Bind the reasoning_gym dataset’s scorer
def make_reward_fn(dataset_for_scoring):
    def rg_reward_fn(prompts, completions, samples, **kwargs):
        rewards = []
        for comp, s in zip(completions, samples):
            ans = extract_answer(comp)
            rewards.append(dataset_for_scoring.score_answer(answer=ans, entry=s["entry"]))
        return rewards
    return rg_reward_fn

reward_fn = make_reward_fn(dataset)  # rg_data has score_answer via its dataset object


def formatting_func(sample):
    # return a single string per sample (GRPO will handle batching)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{sample['question']}"
    )
# lets load the model
MODEL_NAME = "HuggingFaceTB/SmolLM-135M-Instruct"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device="cuda"
        attn_implementation="flash_attention_2",
    )
    return tokenizer, llm

tokenizer, model = load_model()

# Load LoRa config
# LoRa is a technique to fine-tune a model with a small number of parameters, while keeping the original model weights frozen.
# The advantage is that it is much faster and more memory efficient than full fine-tuning.
# Tradeoffs are that LoRa requires more training data and time, and it may not perform as well as full fine-tuning. 
lora_config = LoraConfig(
    r=16, # rank of the LoRA matrix
    lora_alpha=32, # alpha of the LoRA matrix
    target_modules="all-linear", # target modules to apply the LoRA matrix to
    task_type="CAUSAL_LM",
)

# get_peft_model is a function that applies the LoRa matrix to the model.
model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())


# 3) Initialize GRPOConfig
training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    formatting_func=formatting_func,
    reward_funcs=[reward_fn],
)

# Train model
wandb.init(project="GRPO")
trainer.train()




\
"""
device = "mps" if torch.backends.mps.is_available() else "cpu"

from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "HuggingFaceTB/SmolLM-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "mps" else torch.float32
).to(device).eval()

inputs = tokenizer("Hello", return_tensors="pt").to(device)
with torch.inference_mode():
    out = llm.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))
"""

