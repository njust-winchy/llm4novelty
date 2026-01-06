import pandas as pd
import ast
from tqdm import tqdm
import json


example = []
df = pd.read_csv('emnlp23_output-format.csv', encoding='utf-8')
review_novelty = list(df['review_novelty'])
paper_novelty = list(df['paper_novelty'])
introduction_sentence = list(df['introduction_sentence'])
reviews = list(df['reviews'])
output_format = list(df['output_format'])
save_list = []
for idx, items in tqdm(enumerate(review_novelty)):
    save_dic = {}
    id = df['paper_id'][idx]
    title = df['title'][idx]
    abstract = df['abstract'][idx]
    keywords = df['keywords'][idx]
    decision = df['decision'][idx]
    n_p = ast.literal_eval(paper_novelty[idx])
    r = ast.literal_eval(review_novelty[idx])
    if len(n_p) == 0:
        if len(r) != 0:
            example.append(output_format[idx])
            continue
        else:
            continue
    if len(r) == 0:
        continue
    with open(f'retrieve/{id}.json', encoding='utf-8') as f:
        retrieve_data = json.load(f)
    f.close()
    n_r = output_format[idx]
    save_dic['id'] = str(id)
    save_dic['title'] = title
    save_dic['abstract'] = abstract
    save_dic['keywords'] = keywords
    save_dic['decision'] = decision
    save_dic['paper_novelty'] = n_p
    save_dic['review_novelty'] = r
    save_dic['output_format'] = n_r
    save_dic['retrieved_title'] = retrieve_data[0]['title']
    save_dic['retrieved_abstract'] = retrieve_data[0]['abstract']
    save_list.append(save_dic)

with open('Final_data.json', 'w', encoding='utf-8') as f:
    json.dump(save_list, f, ensure_ascii=False, indent=4)










# qwen3-8b as backbone
# rerank review novelty
# 针对每个句子找到最相关的意见, 去除相似冗余的. 作为最后的输出目标.


# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# model_name = "Qwen/Qwen3-4B-Instruct-2507"
#
# # load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
#
# # prepare the model input
# prompt = "Give me a short introduction to large language model."
# messages = [
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#
# # conduct text completion
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=16384
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
#
# content = tokenizer.decode(output_ids, skip_special_tokens=True)
#
# print("content:", content)

