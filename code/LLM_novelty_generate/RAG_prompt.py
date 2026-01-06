# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm


model_list = ["Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
              "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"]

# model_list_api = ['gpt-5', 'gpt-4o', 'Gemini-2.0-Flash']
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

for model_name in model_list:
    save_list = []
    prior = model_name.split('/')[1]
    save_name = prior + '_rag.json'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    for items in tqdm(data):
        prompt = build_novelty_rag_prompt(items['paper_novelty'], items['retrieved_title'], items['retrieved_abstract'])

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
