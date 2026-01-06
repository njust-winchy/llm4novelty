# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm

model_list = ["ECNU-SEA/SEA-E", "ECNU-SEA/SEA-S"]


# model_list_api = ['gpt-5', 'gpt-4o', 'Gemini-2.0-Flash']
def build_novelty_prompt(sentences):

    return f"""
You are a peer-review expert specializing in novelty evaluation of academic papers. 
I will provide sentences from the introduction of a paper that describe its novelty. 
Your task is to evaluate the novelty of the paper and present your evaluation strictly in the following format:

Positive Novelty Evaluations
- 

Neutral Novelty Evaluations
- 

Negative Novelty Evaluations
- 

Rules:
- If the work is highly innovative, add an item under "Positive".
- If the work shows some improvement but is only moderately new, add it under "Neutral".
- If the work lacks substantial novelty, add it under "Negative".
- Each category can contain multiple evaluations or remain empty (with only a dash).
- Keep evaluations concise and academic in tone.

Input:
\"\"\"  
{sentences}  
\"\"\"
"""


with open('Final_data.json', encoding='utf-8') as f:
    data = json.load(f)
f.close()

for model_name in model_list:
    save_list = []
    prior = model_name.split('/')[1]
    save_name = prior + '_rag.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda"
    )
    for items in tqdm(data):
        prompt = build_novelty_prompt(items['paper_novelty'])

        # prepare the model input
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=4096
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        items['output'] = content
        save_list.append(items)

    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, indent=4)
    f.close()
    print(save_name + ' has been saved')
