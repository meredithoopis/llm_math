import os 
import json 
import pandas as pd 
import torch 
from tqdm import tqdm 
import re 
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training) 
from data_processing import process_data, list_formatted 
from config import get_config 
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig) 

#unset top_p 
#Eval accuracy on "cot": 0.8283132530120482
#Eval accuracy on "fs_cot": 0.6054216867469879
def main(): 
    config = get_config() 
    mode = config.evaluate_mode 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    MODEL = f"{config.hf_account}/{config.model_hf_name}" 
    
    lora_config = PeftConfig.from_pretrained(MODEL) 
    bnb_config = BitsAndBytesConfig( load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16 ) 
    model = AutoModelForCausalLM.from_pretrained( lora_config.base_model_name_or_path, return_dict = True, quantization_config = bnb_config, device_map = "auto", trust_remote_code=True ) 
    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path) 
    tokenizer.pad_token = tokenizer.eos_token 
    model = PeftModel.from_pretrained(model, MODEL).to(DEVICE) 
    
    generation_config = model.generation_config 
    generation_config.max_new_tokens = config.max_new_tokens 
    generation_config.temperature = config.temperature 
    #generation_config.top_p = config.top_p 
    generation_config.num_return_sequences = config.num_return_sequences 
    generation_config.pad_token_id = tokenizer.eos_token_id 
    generation_config.eos_token_id = tokenizer.eos_token_id 
    
    if mode == "cot": 
        test_samples = process_data(config.dataset_test, tokenizer, mode="cot") 
    elif mode == "fs_cot": 
        test_samples = process_data(config.dataset_test, tokenizer, mode="fs_cot") 
    
    results = [] 
    for problem in tqdm(test_samples, desc = "Generating results"): 
        encoded_qs = tokenizer(problem, return_tensors="pt").to(DEVICE) 
        with torch.inference_mode(): 
            outputs = model.generate( input_ids = encoded_qs.input_ids, attention_mask = encoded_qs.attention_mask, generation_config = generation_config ) 
        ans = tokenizer.decode(outputs[0], skip_special_tokens=True) 
        res = re.findall(r'\\boxed\{(.*)\}', ans)[-1] 
        if res == "": res = 'E' 
        results.append(res) 
    
    dick = list_formatted(config.dataset_test) 
    list_id = dick['list_id'] 
    list_questions = dick['list_questions'] 
    As = dick['list_A'] 
    Bs = dick['list_B'] 
    Cs = dick['list_C'] 
    Ds = dick['list_D'] 
    answers = dick['list_answers'] 
    df_test = pd.DataFrame(list(zip(list_id, list_questions, As, Bs, Cs, Ds, answers, results)), columns=['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'result']) 
    correct = (df_test['answer'] == df_test['result']).sum() 
    total = len(df_test) 
    accuracy = correct / total 
    print("Accuracy:", accuracy) 
    
if __name__ == "__main__": 
    main()