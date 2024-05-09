def tokenize_prompt(tokenizer, qs, choices, explanation, answer): 
    prompt = gen_prompt_train(qs,choices, explanation, answer)
    tokenized_prompt = tokenizer(
          prompt, padding=True, truncation=True
     )
    return tokenized_prompt


def gen_prompt_train(qs, choices, explanation, answer): 
    answer_char = answer.split('. ')[0] 
    formatted_answer = f"$\\boxed{{{answer_char}}}$" 

    return f"""
<|im_start|>system 
You are a Math expert. You will receive multiple choice math questions with options, solve them step by step if possible and 
choose the correct option.

<|im_start|>user
### Question: 
{qs}
### Choices: 
{choices}\nProvide your reasons step by step, and put your final answer within \\boxed{{}}. 
### Answer: 

<|im_start|>assistant
{explanation}\nThe answer is{formatted_answer}.

""".strip()

def gen_prompt_test(qs,choices): 

    return f"""
<|im_start|>system 
You are a Math expert. You will receive multiple choice math questions with options, solve them step by step if possible and 
choose the correct option.

<|im_start|>user
### Question: 
{qs}
### Choices: 
{choices}\nProvide your reasons step by step, and put your final answer within \\boxed{{}}. 
### Answer: 

<|im_start|>assistant
""".strip()

def gen_few_shot_prompt(qs, choices): 
        return f"""
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
{qs}
### Choices: 
{choices}\nProvide your reasons step by step, and put your final answer within \\boxed{{}}. 
### Answer: 

<|im_start|>assistant


""".strip()


def num_params(model):
    #Print the number of trainable parameters of the model 
    total_params = 0 
    trainable_params = 0 
    for _, param in model.named_parameters(): 
         total_params += param.numel()
         if param.requires_grad: 
              trainable_params += param.numel()
    
    print(f"Trainable params: {trainable_params} || all params: {total_params} || trainables: {100 * trainable_params/ total_params}")