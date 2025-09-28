GRPO + Reasoning Gym (Smol LLM)
================================

Train and evaluate a small open model (SmolLM-135M) on algorithmically verifiable tasks from Reasoning Gym. Project uses uv for dependency management and includes local scripts plus a starter notebook.

Requirements
------------
- macOS or Linux
- Python 3.12/3.13 (project uses uv)
- For GPU training (optional): CUDA GPU (e.g., A100 on Colab)

Quickstart (uv)
---------------
1) Clone and enter the project directory.
2) Create/activate your env (uv manages venvs automatically). If you already use a custom venv like `.grpo`, pass `--active`.

Install core deps:
```bash
uv sync --active  # installs from pyproject/uv.lock
```

If you need extras (examples):
```bash
uv add --active transformers torch reasoning-gym rich trl peft accelerate datasets wandb
```

Environment Keys
----------------
- OpenAI (optional, only for `intro_reasoning_gym_.py` if you test OpenAI): set `OPENAI_API_KEY` in your shell or `.env`.
- Hugging Face (for gated models/datasets):
```bash
uv run --active huggingface-cli login
```

Run Local Scripts
-----------------
- Smol model + Reasoning Gym demo:
```bash
uv run --active python smol_llm.py
```

- RL with GRPOTrainer (Reasoning Gym rewards):
```bash
uv run --active python rl_smol.py
```

Jupyter Notebook
----------------
Install and launch:
```bash
uv add --active jupyter ipykernel
uv run --active python -m ipykernel install --user --name grpo --display-name "Python (.grpo)"
uv run --active jupyter notebook
```
Open `notebooks/intro_reasoning_gym.ipynb` and select the "Python (.grpo)" kernel.

Reasoning Gym Basics
--------------------
Create and iterate a dataset:
```python
import reasoning_gym
dataset = reasoning_gym.create_dataset("propositional_logic", size=5, seed=42)
for entry in dataset:
    print(entry["question"])  # input
    reward = dataset.score_answer(answer=entry["metadata"].get("example_answer"), entry=entry)
    print(reward)
```

GRPO Wiring (Summary)
---------------------
- Prompts are built by `formatting_func(sample) -> str` and should enforce:
  - Think in `<think>...</think>` and output final in `<answer>...</answer>`.
- Reward function extracts `<answer>` and calls `dataset.score_answer(answer, entry)`.
- `GRPOTrainer` handles generations per prompt and calls your reward function.

Colab (A100) Tips
-----------------
Installs:
```bash
!pip -q install trl==0.14.0 transformers==4.47.1 accelerate==1.2.1 \
               datasets==3.2.0 peft==0.14.0 wandb==0.19.7 reasoning-gym \
               bitsandbytes==0.45.2
```
Model/device:
```python
import torch
device = "cuda"
dtype = torch.bfloat16
```
Set `bf16=True` in `GRPOConfig` and `optim="adamw_8bit"` if using bitsandbytes.

Troubleshooting
---------------
- Model gated: run `huggingface-cli login` and accept model terms on its HF page.
- Mac M-series: use `device="mps"` and `torch.float16`; skip bitsandbytes/flash-attn.
- Python 3.13 Torch issues: prefer Python 3.12 for best compatibility.

License
-------
MIT (project code). Check individual model/dataset licenses on Hugging Face.

