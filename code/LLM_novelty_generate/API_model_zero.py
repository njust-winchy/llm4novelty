import torch
from openai import OpenAI
from tqdm import tqdm
import json
import os

model_list_api = ['gpt-5', 'gpt-4o', 'Gemini-2.0-Flash']



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

def gpt4task(prompt, model):
    client = OpenAI(
        base_url='https://api.kksj.org/v1',
        # required but ignored
        api_key='sk-yqwpNJSFxMqyEDjaJockxdXNQBkCMiihZdL2DPlCF9kCGWGY',
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,

            }
        ],
        model=model
    )
    return chat_completion.choices[0].message.content


for model_name in model_list_api:
    save_list = []

    save_name = model_name + '_zero.json'

    # === 1. Load existing results if available ===
    if os.path.exists(save_name):
        with open(save_name, 'r', encoding='utf-8') as f:
            save_list = json.load(f)
        processed_ids = {item['id'] for item in save_list}
        print(f"? Resuming from {save_name}: {len(processed_ids)} items already processed.")
    else:
        processed_ids = set()
        print(f"? Starting new run: {save_name}")

    # === 2. Continue processing unprocessed items ===
    for items in tqdm(data, desc=f"Processing {model_name}"):
        if items['id'] in processed_ids:
            continue  # Skip already processed items

        try:
            prompt = build_novelty_prompt(items['paper_novelty'])
            # === Run the model ===
            content = gpt4task(prompt, model_name)
            items['output'] = content
            save_list.append(items)
            processed_ids.add(items['id'])
            # === Save progress after each item ===
            with open(save_name, 'w', encoding='utf-8') as f:
                json.dump(save_list, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"? Error processing {items.get('paper_novelty', 'unknown')}: {e}")
            continue  # Skip on error and move on

    print(f"? Finished and saved {save_name}: {len(save_list)} total items.")
