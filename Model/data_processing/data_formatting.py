import os
import json
from datasets import Dataset
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel

data_folder = "../data/code-optimization"
data_json_path = os.path.join(data_folder, "data.json")
output_path = os.path.join(data_folder, "data_formatted.json")

with open(data_json_path, "r") as file:
    data = json.load(file)
    
dataset = Dataset.from_dict({"conversations": data})

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = None,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3",  # Specify the desired chat template
    mapping={"role": "from_agent", "content": "value_msg", "user": "querry", "assistant": "response"}
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo["conversations"], tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

dataset.to_json(output_path)

