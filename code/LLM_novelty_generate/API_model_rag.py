from openai import OpenAI
from tqdm import tqdm
import json
import os

model_list_api = ['gpt-5', 'gpt-4o', 'Gemini-2.0-Flash']


def build_novelty_rag_prompt(sentences, retrieve_title, retrieve_abstract):
    return f"""You are a peer-review expert specializing in novelty evaluation of academic papers.
    I will provide:
    1. Sentences from the introduction of a paper that describe its novelty.
    2. Titles and abstracts of 5 related papers retrieved from a local literature database.

    Your task is to evaluate the novelty of the given paper by comparing it against the retrieved references.
    Output your evaluation strictly in the following format:

    Positive Novelty Evaluations
    -

    Neutral Novelty Evaluations
    -

    Negative Novelty Evaluations
    -

    Rules:
    - If the contribution introduces a fundamentally new idea not present in the retrieved references, write it under "Positive".
    - If the contribution is similar to but not exactly the same as retrieved references, write it under "Neutral".
    - If the claimed contribution overlaps strongly with retrieved references and lacks originality, write it under "Negative".
    - Each section can be empty if not applicable (just keep the dash).
    - Keep evaluations concise, academic in tone, and grounded in the retrieved references.

    [Input Sentences]
    {sentences}

    [Retrieved References]
    1. Title: {retrieve_title[0]}
       Abstract: {retrieve_abstract[0]}

    2. Title: {retrieve_title[1]}
       Abstract: {retrieve_abstract[1]}

    3. Title: {retrieve_title[2]}
       Abstract: {retrieve_abstract[2]}

    4. Title: {retrieve_title[3]}
       Abstract: {retrieve_abstract[3]}

    5. Title: {retrieve_title[4]}
       Abstract: {retrieve_abstract[4]}
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

    save_name = model_name + '_rag.json'

    # === 1. Load existing results if available ===
    if os.path.exists(save_name):
        with open(save_name, 'r', encoding='utf-8') as f:
            save_list = json.load(f)
        processed_ids = {item['paper_novelty'] for item in save_list}
        print(f"? Resuming from {save_name}: {len(processed_ids)} items already processed.")
    else:
        processed_ids = set()
        print(f"? Starting new run: {save_name}")

    # === 2. Continue processing unprocessed items ===
    for items in tqdm(data, desc=f"Processing {model_name}"):
        if items['paper_novelty'] in processed_ids:
            continue  # Skip already processed items

        try:
            prompt = build_novelty_rag_prompt(items['paper_novelty'], items['retrieved_title'], items['retrieved_abstract'])
            # === Run the model ===
            content = gpt4task(prompt, model_name)
            items['output'] = content
            save_list.append(items)
            processed_ids.add(items['paper_novelty'])

            # === Save progress after each item ===
            with open(save_name, 'w', encoding='utf-8') as f:
                json.dump(save_list, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(f"? Error processing {items.get('paper_novelty', 'unknown')}: {e}")
            continue  # Skip on error and move on

    print(f"? Finished and saved {save_name}: {len(save_list)} total items.")
