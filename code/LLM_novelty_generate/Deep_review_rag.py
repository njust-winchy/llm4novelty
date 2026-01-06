from ai_researcher.deep_reviewer import DeepReviewer
import json
import torch
from tqdm import tqdm

def build_novelty_rag_prompt(retrieve_title, retrieve_abstract):
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
model_size = ["7B", "14B"]
# Initialize DeepReviewer
with open('/data/scratch/qc25257/datasets/Final_data.json', encoding='utf-8') as f:
    data = json.load(f)
f.close()
for model in model_size:
    save_list = []
    save_name = 'DeepReviewer_' + model + '_rag.json'
    reviewer = DeepReviewer(
        model_size=model,  # Use "7B" for the smaller model
        device="cuda",
        tensor_parallel_size=1,  # Increase for multi-GPU setup
        gpu_memory_utilization=0.95
    )
    for items in tqdm(data):
        novelty_prompt = build_novelty_rag_prompt(items['retrieved_title'], items['retrieved_abstract'])
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
        standard_list = []
        for rev in fast_review:
            standard_list.append(rev['raw_text'])
        #print(standard_review)
        items['fast_output'] = fast_list
        items['standard_output'] = standard_list
        save_list.append(items)

    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, indent=4)
    f.close()
    print(save_name + ' has been saved')

