from ai_researcher.cycle_reviewer import CycleReviewer
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
model_size = ["8B", "70B"]
model_dir = ["/data/scratch/qc25257/local_model/CycleReviewer-ML-Llama-3.1-8B/", "/data/scratch/qc25257/local_model/CycleReviewer-Llama-3.1-70B/"]

# Initialize DeepReviewer
with open('Final_data.json', encoding='utf-8') as f:
    data = json.load(f)
f.close()

for idx, model in enumerate(model_size):
    save_list = []
    save_name = 'CycleReviewer_' + model + '_rag.json'
    reviewer = CycleReviewer(
        model_size=model,  # Use "7B" for the smaller model
        custom_model_name=model_dir[idx],
        device="cuda",
    )
    for items in tqdm(data):
        novelty_prompt = build_novelty_rag_prompt(items['retrieved_title'], items['retrieved_abstract'])
        # Load paper content
        paper_content = items['paper_novelty']# Replace with actual paper content

        # Generate reviews in different modes
        # Fast Mode for quick overview
        fast_review = reviewer.evaluate(paper_content, custom_prompt=novelty_prompt)


        items['output'] = fast_review
        save_list.append(items)

    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(save_list, f, indent=4)
    f.close()
    print(save_name + ' has been saved')

