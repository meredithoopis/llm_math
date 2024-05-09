import os
import json
import pandas as pd
import torch
import tqdm
import re
from config import get_config
from peft import (LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training)
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

def main(): 
    config = get_config()
    mode = config.inference_mode
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL = f"{config.hf_account}/{config.model_hf_name}"

    lora_config = PeftConfig.from_pretrained(MODEL) 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        lora_config.base_model_name_or_path, 
        return_dict = True, 
        quantization_config = bnb_config, 
        device_map = "auto", 
        trust_remote_code=True 
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token 
    
    model = PeftModel.from_pretrained(model, MODEL).to(DEVICE)
    
    generation_config = model.generation_config
    generation_config.max_new_tokens = config.max_new_tokens
    generation_config.temperature = config.temperature
    generation_config.top_p = config.top_p
    generation_config.num_return_sequences = config.num_return_sequences
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    
    if mode == "cot": 
        prompt = f"""
    <|im_start|>system
    You are a Math expert. You will be given multiple choice math questions, solve them step by step if available and choose the correct answer. 

    <|im_start|>user
    ### Question:
    Một cửa hàng bán xe đạp với giá gốc là 1.500.000 đồng. Cửa hàng quyết định giảm giá 20% cho khách hàng mua trực tiếp tại cửa hàng và giảm thêm 5% nếu khách hàng thanh toán bằng thẻ ngân hàng. Hỏi giá bán cuối cùng của chiếc xe đạp nếu một khách hàng mua trực tiếp tại cửa hàng và thanh toán bằng thẻ ngân hàng là bao nhiêu?
    Các lựa chọn:
    A. 1.125.000 đồng
    B. 1.140.000 đồng
    C. 1.155.000 đồng
    D. 1.170.000 đồng
    Hãy giải thích từng bước và đưa câu trả lời cuối cùng vào \\boxed{{}}.

    ### Answer:
    <|im_start|>assistant
    """.strip()
    
    if mode == "fs_cot": 

        prompt = """
Below are some examples of questions and their correct responses. After the examples, write a suitable response that suits the new question. 

### Exmaple 1:
<|im_start|>user
### Question:
Tìm giá trị của x trong phương trình 3x + 2 = 11.
### Choices:
A. x = 1
B. x = 2
C. x = 3
D. x = 4
Explain step by step and put your final answer within \\boxed{{}}.
### Answer:


<|im_start|>assistant
Để tìm giá trị của x, chúng ta cần tách biệt x về một vế của phương trình bằng cách thực hiện các phép tính nghịch đảo.
Phương trình cho: 3x + 2 = 11
Bước 1: Trừ 2 cho cả hai vế để tách biệt hạng chứa x.
3x + 2 - 2 = 11 - 2
3x = 9
Bước 2: Chia cả hai vế cho 3 để tìm giá trị của x.
3x/3 = 9/3
x = 3
Vì vậy, giá trị của x trong phương trình 3x + 2 = 11 là 3.
The answer is $\\boxed{{C}}$.

### Example 2:
<|im_start|>user
### Question:
Tính diện tích của một hình tròn có bán kính 5 cm?
### Choices:
A. 10π cm²
B. 25π cm²
C. 50π cm²
D. 100π cm²
Explain step by step and put your final answer within \\boxed{{}}.
### Answer:

<|im_start|>assistant
Diện tích của một hình tròn được tính theo công thức:
Diện tích = πr²
Trong đó r là bán kính của hình tròn.
Cho:
Bán kính (r) = 5 cm
Thay giá trị của r vào công thức:
Diện tích = π × (5 cm)²
Diện tích = π × 25 cm²
Diện tích = 25π cm²
Vì vậy, diện tích của một hình tròn có bán kính 5 cm là 25π cm².
Câu trả lời là $\\boxed{{B}}$.

### New task: 
<|im_start|>system 
You are a Math expert. You will receive multiple choice math questions with options, solve them step by step if possible and 
choose the correct option.

<|im_start|>user
### Question: 
Giá trị của chữ số 8 trong số thập phân 50,289:
### Choices: 
A. 8
B. \\frac{8}{10}
C. \\frac{8}{100}
D. 80
Provide your reasons step by step, and put your final answer within \\boxed{{}}. 
### Answer: 

<|im_start|>assistant""".strip()
    encoding = tokenizer(prompt, return_tensors = "pt").to(DEVICE)
    with torch.inference_mode(): 
        outputs = model.generate(
            input_ids = encoding.input_ids, 
            attention_mask = encoding.attention_mask, 
            generation_config = generation_config
        )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__": 
    main()