# -*- coding: utf-8 -*-

from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
from tqdm import tqdm


model_list = ["Qwen/Qwen3-32B", "Qwen/Qwen3-14B", "Qwen/Qwen3-8B", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
              "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]
#
# model_list_api = ['gpt-5', 'gpt-4o', 'Gemini-2.0-Flash']

def build_novelty_prompt(sentences, input_1, input_2, output_1, output_2):

    return f"""You are a peer-review expert specializing in novelty evaluation of academic papers. 
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

### Example 1
Input:
{input_1}
Output:
{output_1}

### Example 2
Input:
{input_2}
Output:
{output_2}

### Now it’s your turn:
Input:
{sentences}
 """


with open('Final_data.json', encoding='utf-8') as f:
    data = json.load(f)
f.close()


def get_few_shot_examples(dataset, current_sample, k=2):
    candidates = [x for x in dataset if x['paper_novelty'] != current_sample]
    few_shot_examples = random.sample(candidates, k)
    return few_shot_examples



for model_name in model_list:
    save_list = []
    prior = model_name.split('/')[1]
    save_name = prior + '_few_shot.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    for items in tqdm(data):
        few_shot_examples = get_few_shot_examples(data, items['paper_novelty'])
        prompt = build_novelty_prompt(items['paper_novelty'], few_shot_examples[0]['paper_novelty'], few_shot_examples[1]['paper_novelty'], few_shot_examples[0]['output_format'], few_shot_examples[1]['output_format'])
        # prepare the model input


        messages = [
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
    print(save_name+' has been saved')
