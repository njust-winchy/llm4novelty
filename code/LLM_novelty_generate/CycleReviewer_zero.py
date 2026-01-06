from ai_researcher import CycleReviewer
import json
import random
import torch


from tqdm import tqdm

def build_novelty_prompt():

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

"""
model_size = ["8B", "70B"]
model_dir = ["/data/scratch/qc25257/local_model/CycleReviewer-ML-Llama-3.1-8B/", "/data/scratch/qc25257/local_model/CycleReviewer-Llama-3.1-70B/"]
# Initialize DeepReviewer
with open('Final_data.json', encoding='utf-8') as f:
    data = json.load(f)
f.close()

model_id = "WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B"

for idx, model in enumerate(model_size):
    save_list = []
    save_name = 'CycleReviewer_' + model + '_zero.json'
    reviewer = CycleReviewer(
        model_size=model,  # Use "7B" for the smaller model
        custom_model_name=model_dir[idx],
        device="cuda"
    )
    for items in tqdm(data):
        novelty_prompt = build_novelty_prompt()
        # Load paper content
        paper_content = items['paper_novelty']# Replace with actual paper content

        # Generate reviews in different modes
        # Fast Mode for quick overview
        fast_review = reviewer.evaluate(paper_content, custom_prompt=novelty_prompt)
        fast_list = []
        for rev in fast_review:
            fast_list.append(rev['raw_text'])
        print()
        #print(standard_review)
        items['fast_output'] = fast_list
        save_list.append(items)
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, indent=4)
    f.close()
    print(save_name+' has been saved')


