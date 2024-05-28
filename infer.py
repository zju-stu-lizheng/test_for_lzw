import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraModel, get_peft_model

model_id = "NousResearch/Llama-2-7b-chat-hf"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": torch.cuda.current_device()},
    trust_remote_code=True,
    use_auth_token=True,
)

peft_model = PeftModel.from_pretrained(base_model, "output_dir/checkpoint-2840")
