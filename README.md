# Finetuning LLM for Elementary Math

## Overview
This project aims to fine-tune a large language model (LLM) to perform elementary mathematics tasks. The goal is to enhance the model's capability to solve basic arithmetic problems, understand elementary math concepts, and provide explanations for its solutions. The targeted audience includes educators, students, and developers interested in educational AI applications.

data: [ZaloAI](https://challenge.zalo.ai/portal/elementary-maths-solving)


## Installation
To get started, clone the repository and install the required dependencies:

```
git clone https://github.com/meredithoopis/llm_math.git
```

```
cd llm_math 
```

Install necessary dependencies: 
```
pip install -r requirements.txt
``` 

## Dataset
The model was evaluated using the dataset of ZaloAI 2023 challenge: [ZaloAI](https://challenge.zalo.ai/portal/elementary-maths-solving)

## Usage 
To train the model, run and adjust the hyperparameters as need: 
```
python main.py 
```

Evaluation: 
```
python evaluate.py 
```

Inference: 
```
python inference.py 
```
