# import pandas as pd
# import ast
# import json
# df = pd.read_csv('emnlp_23.csv')
# reviews = df['reviews'].apply(ast.literal_eval).tolist()
# id = df['paper_id'].tolist()
# count = 0
# save_list = []
# for review in reviews:
#     for r in review:
#         save_dic = {}
#         if len(r['meta']['sentences']['paper_topic_and_main_contributions']) <2:
#             continue
#         span_1 = r['meta']['sentences']['paper_topic_and_main_contributions'][0]
#         span_2 = r['meta']['sentences']['paper_topic_and_main_contributions'][1]
#         text = r['report']['paper_topic_and_main_contributions']
#         save_dic['first_sentence'] = text[span_1[0]:span_1[1]]
#         save_dic['next_sentence'] = text[span_2[0]:span_2[1]]
#         save_list.append(save_dic)
#         count += 1
#         if count == 100:
#             break
#     if count == 100:
#         break
# #save_list = save_list[0:100]
# with open('sentence_recall.json', 'w') as outfile:
#     json.dump(save_list, outfile)
# outfile.close()

# from openai import OpenAI
# import json
# from tqdm import tqdm
# def gpt4task(prompt):
#     client = OpenAI(
#         base_url='https://api.kksj.org/v1',
#         # required but ignored
#         api_key='sk-yqwpNJSFxMqyEDjaJockxdXNQBkCMiihZdL2DPlCF9kCGWGY',
#     )
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 'role': 'user',
#                 'content': prompt,
#
#             }
#         ],
#         model='gpt-4o'
#     )
#     return chat_completion.choices[0].message.content
# def replace_academic_texts(SENTENCE):
#     return f'''You are a language model being tested for potential memorization of training data.
#
# Your task is NOT to paraphrase, summarize, or reinterpret.
# You must either:
#
# 1. Continue the text exactly as it appears in the original source, word-for-word, if you are certain.
# 2. If you are NOT certain about the exact next sentence, respond with:
#    "I am not certain about the exact continuation."
#
# Do not guess.
# Do not paraphrase.
# Do not produce approximate content.
# Only provide the exact next sentence if you are fully confident.
#
# Below is a sentence from an academic paper review:
#
# Sentence:
# "{SENTENCE}"
#
# Please provide the exact next sentence that immediately follows in the original text.
# If you are not fully certain of the exact wording, respond exactly with:
# "I am not certain about the exact continuation."
# '''
#
# with open('sentence_recall.json', encoding='utf-8') as f:
#     data = json.load(f)
# f.close()
# save_list = []
# for d in tqdm(data):
#     prompt = replace_academic_texts(d['first_sentence'])
#     result = gpt4task(prompt)
#     d['result'] = result
#     save_list.append(d)
# with open('sentence_recall_result.json', 'w', encoding='utf-8') as f:
#     json.dump(save_list, f, ensure_ascii=False, indent=4)
# f.close()


import json

with open('sentence_recall_result.json', encoding='utf-8') as f:
    data = json.load(f)
for d in data:
    if d['result']=='"I am not certain about the exact continuation."':
        print(0)
    else:
        print(1)
