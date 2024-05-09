import json
import os
import torch 
import torch.nn as nn 
import transformers 
from utils import (num_params, tokenize_prompt, gen_prompt_train)
from train import train 
import bitsandbytes as bnb 
from data_processing import process_data
from config import get_config
from pprint import pprint 
from tqdm import tqdm 
from datasets import load_dataset, Dataset 
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training)
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

#https://huggingface.co/lisagrace/deepseek-math-7b-viet


if __name__ == "__main__": 
    config = get_config()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True, 
        bnb_4bit_use_double_quant= True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model, 
        device_map= "auto", 
        trust_remote_code=True, 
        quantization_config = bnb_config 
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.pad_token = tokenizer.eos_token
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
    r = config.lora_r, 
    lora_alpha = config.lora_alpha, 
            target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"
    ],
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    num_params(model)

    generation_config = model.generation_config
    generation_config.max_new_tokens = config.max_new_tokens
    generation_config.temperature = config.temperature
    generation_config.top_p = config.top_p
    generation_config.num_return_sequences = config.num_return_sequences
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    choices_data = process_data(config.dataset_train, tokenizer, mode="train")
    train(model, tokenizer, choices_data)
    MODEL = f"{config.hf_account}/{config.model_hf_name}"
    model.save_pretrained(config.model_hf_name)
    model.push_to_hub(MODEL, use_auth_token = True)
    tokenizer.save_pretrained(config.model_hf_name)
    tokenizer.push_to_hub(MODEL, use_auth_token = True)
    #todo: Maybe save model 
