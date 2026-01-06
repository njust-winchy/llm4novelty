from ai_researcher.deep_reviewer import DeepReviewer
import json
import random
import torch
from tqdm import tqdm

def build_novelty_prompt(input_1, input_2, output_1, output_2):

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

### Now it¡¯s your turn:

 """
model_size = ["7B", "14B"]
# Initialize DeepReviewer
with open('/data/scratch/qc25257/datasets/Final_data.json', encoding='utf-8') as f:
    data = json.load(f)
f.close()
def get_few_shot_examples(dataset, current_sample, k=2):
    candidates = [x for x in dataset if x['paper_novelty'] != current_sample]
    few_shot_examples = random.sample(candidates, k)
    return few_shot_examples

for model in model_size:
    save_list = []
    save_name = 'DeepReviewer_' + model + '_few.json'
    reviewer = DeepReviewer(
        model_size=model,  # Use "7B" for the smaller model
        device="cuda",
        tensor_parallel_size=1,  # Increase for multi-GPU setup
        gpu_memory_utilization=0.95
    )
    for items in tqdm(data):
        few_shot_examples = get_few_shot_examples(data, items['paper_novelty'])
        novelty_prompt = build_novelty_prompt(items['paper_novelty'], few_shot_examples[0]['paper_novelty'],
                                      few_shot_examples[1]['paper_novelty'], few_shot_examples[0]['output_format'],
                                      few_shot_examples[1]['output_format'])
        # Load paper content
        paper_content = items['paper_novelty']# Replace with actual paper content

        # Generate reviews in different modes
        # Fast Mode for quick overview
        fast_review = reviewer.evaluate([paper_content], mode="Fast Mode", custom_prompt=novelty_prompt)
        fast_list = []
        for rev in fast_review:
            fast_list.append(rev['raw_text'])
        #print(fast_review)
        # Standard Mode with multiple reviewers
        standard_review = reviewer.evaluate([paper_content], mode="Standard Mode", reviewer_num=3, custom_prompt=novelty_prompt)
        #print(standard_review)
        standard_list = []
        for rev in fast_review:
            standard_list.append(rev['raw_text'])
        # print(standard_review)
        items['fast_output'] = fast_list
        items['standard_output'] = standard_list
        save_list.append(items)
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, indent=4)
    f.close()
    print(save_name + ' has been saved')

