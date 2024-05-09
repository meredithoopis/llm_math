import tqdm 
import json 
from utils import (tokenize_prompt, gen_prompt_test, gen_prompt_train, gen_few_shot_prompt)
from datasets import load_dataset, Dataset 
import ast 


def process_data(filepath, tokenizer, mode="train"):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if mode == "train":         
        training_data = []
        for sample in data["data"]: 
            try: 
                choices = ast.literal_eval(sample['choices'])
            except: break 
            explanation = sample['explanation'].strip()
            qs = sample['question']
            answer = sample['answer']
            choices = '\n'.join(choices)
            training_sample = tokenize_prompt(
                tokenizer, qs, choices, explanation, answer
            )
            training_data.append(training_sample)
            choices_data = Dataset.from_list(training_data)
            
        return choices_data 

    if mode == "cot": #chain-of-thoughts 
        test_data = []
        for sample in data["data"]: 
            try: 
                choices = sample["choices"]
            except: break 
            qs = sample['question']
            choices = '\n'.join(choices)
            test_sample = gen_prompt_test(qs, choices)
            test_data.append(test_sample)
        return test_data

    if mode == "fs_cot": #few-shot chain of thoughts 
        test_samples = []
        for sample in data["data"]: 
            try: 
                choices = sample['choices']
            except: break 
            qs = sample['question']
            choices = '\n'.join(choices)
            test_sample = gen_few_shot_prompt(qs,choices)
            test_samples.append(test_sample)
        return test_samples

def list_formatted(filepath): 
    with open(filepath, 'r') as f: 
        data = json.load(f) 
    list_id = []
    list_qs = []
    As = []
    Bs = []
    Cs = []
    Ds = []
    answers = []
    for record in data['data']: 
        id = record['id']
        qs = record['question']
        choices = record['choices']
        answer = record['answer'][0]

        As.append(choices[0])
        Bs.append(choices[1])
        Cs.append(choices[2])
        try: 
            Ds.append(choices[3])
        except IndexError: 
            Ds.append("None")
        list_id.append(id)
        list_qs.append(qs)
        answers.append(answer)
    
    return {
        "list_id": list_id, 
        "list_questions": list_qs, 
        "list_A": As, 
        "list_B": Bs, 
        "list_C": Cs, 
        "list_D": Ds, 
        "list_answers": answers
    }
