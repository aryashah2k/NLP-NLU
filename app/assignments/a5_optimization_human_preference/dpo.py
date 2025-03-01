# %% [markdown]
# # [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)](https://arxiv.org/pdf/2305.18290.pdf)
# 
# ### Reference Code 
# - https://huggingface.co/docs/trl/main/en/dpo_trainer
# - https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py

# %% [markdown]
# Therefore the final dataset object should contain these 3 entries if you use the default DPODataCollatorWithPadding data collator. 
# 
# The entries should be named:
# - prompt
# - chosen
# - rejected

# %%
import os
import torch
# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.environ['http_proxy']  = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
dpo_dataset_dict = {
    "prompt": [
        "hello",
        "how are you",
        "What is your name?",
        "What is your name?",
        "Which is the best programming language?",
        "Which is the best programming language?",
        "Which is the best programming language?",
    ],
    "chosen": [
        "hi nice to meet you",
        "I am fine",
        "My name is Mary",
        "My name is Mary",
        "Python",
        "Python",
        "Java",
    ],
    "rejected": [
        "leave me alone",
        "I am not fine",
        "Whats it to you?",
        "I dont have a name",
        "Javascript",
        "C++",
        "C++",
    ],
}

# %%
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments
)

from typing import Dict, Optional
from trl import DPOTrainer

# %% [markdown]
# # 1. load a pretrained model and tokenizer

# %%
model_name_or_path = "gpt2"
ignore_bias_buffers = False

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
if ignore_bias_buffers:
    # torch distributed hack
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]

model_ref = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# The DPO trainer expects a model of AutoModelForCausalLM, compared to PPO that expects AutoModelForCausalLMWithValueHead for the value function.

# %% [markdown]
# ## 2. Load the Anthropic Helpful-Harmless dataset

# %%
def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]

def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """

    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)

# %%
sanity_check = True
train_dataset = get_hh("train", sanity_check=sanity_check)
eval_dataset = get_hh("test", sanity_check=sanity_check)

# %%
train_dataset

# %%
eval_dataset

# %% [markdown]
# # 3. initialize training arguments:

# %%
learning_rate = 1e-3
per_device_train_batch_size = 8
gradient_accumulation_steps = 1
max_length= 512 
max_prompt_length = 128 
max_target_length =128 
label_pad_token_id = 100
max_steps = 1000
# instrumentation
sanity_check = True
report_to = None
gradient_checkpointing = None
beta = 0.1

# %%
training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    max_steps=max_steps,
    remove_unused_columns=False,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    evaluation_strategy="steps",
    logging_first_step=True,
    logging_steps=5,  # match results in blog post
    eval_steps=500,
    output_dir="./test",
    optim="rmsprop",
    warmup_steps=150,
    report_to=report_to,
    bf16=True,
    gradient_checkpointing=gradient_checkpointing,
    # TODO: uncomment that on the next transformers release
    # gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
)

# %% [markdown]
# # 4. initialize the DPO trainer

# %%
dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=beta,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_length=max_length,
    max_target_length=max_target_length,
    max_prompt_length=max_prompt_length,
    generate_during_eval=True,
)

# %% [markdown]
# # 5. Train

# %%
dpo_trainer.train()

# %%



