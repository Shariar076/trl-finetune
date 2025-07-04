import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
model_name="./output/01-ai/Yi-1.5-9B-Chat"
adapter_path = "./output/yi-1.5-9b-consistent-20250702-013917"
output_dir= f"./output/{model_name.split('/')[-1]}-Consistency-FT"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    # device_map=device_map,
)
tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name)

model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# model.push_to_hub( new_model, new_model+'_local', private=True)
# tokenizer.push_to_hub(new_model, new_model+'_local', private=True)