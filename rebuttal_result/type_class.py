# -*- coding: utf-8 -*-

# from openai import OpenAI
#
# import json
#
# def gpt4task(prompt):
#     client = OpenAI(
#         base_url='',
#         # required but ignored
#         api_key='',
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
# def replace_academic_texts(TITLE, ABSTRACT):
#     return f'''You are an expert NLP conference reviewer.
#
# Your task is to classify the following paper into ONE of the two categories:
#
# 1. Methodological paper
#    - Proposes a new model, algorithm, architecture, training method, optimization technique, or theoretical framework.
#    - Focuses on improving performance, efficiency, or methodology.
#    - The main contribution is a method or technical innovation.
#
# 2. Resource paper
#    - Introduces a dataset, benchmark, shared task, annotation scheme, corpus, evaluation framework, or infrastructure.
#    - The primary contribution is a data resource or evaluation platform.
#    - May include baseline models, but the core contribution is the resource itself.
#
# Important:
# - If the paper proposes both a method and a dataset, classify based on the PRIMARY contribution emphasized in the abstract.
# - Output ONLY one word:
#   - "Methodological"
#   - or "Resource"
# - Do not output any explanation.
#
# Now classify the following paper:
#
# Title:
# \"\"\"
# {TITLE}
# \"\"\"
# Abstract:
# \"\"\"
# {ABSTRACT}
# \"\"\"
# '''
#
# with open('Final_data.json', encoding='utf-8') as f:
#     data = json.load(f)
# f.close()
# save_list = []
# for i in data:
#     title = i['title']
#     abstract = i['abstract']
#     prompt = replace_academic_texts(title, abstract)
#     result = gpt4task(prompt)
#     i['classification'] = result
#     save_list.append(i)
#
# with open('Final_data_classification.json', 'w', encoding='utf-8') as f:
#     json.dump(save_list, f, ensure_ascii=False, indent=4)
# f.close()

import json
import os
import math
def average_cal(cal_list_1):
    rel_score = []
    corr_score = []
    cover_score = []
    clarity_score = []
    for idx, i in enumerate(cal_list_1):
        res = i['Model']
        rel = res['Relevance']
        correct = res['Correctness']['distribution_accuracy']
        cover = res['Coverage']
        clarity = res['Clarity']
        rel_score.append(rel)
        corr_score.append(correct)
        cover_score.append(cover)
        clarity_score.append(clarity)
    rel_score = sum(rel_score) / len(rel_score)
    corr_score = sum(corr_score) / len(corr_score)
    cover_score = sum(cover_score) / len(cover_score)
    valid_scores = [x for x in clarity_score if not math.isnan(x)]
    clarity_score = sum(valid_scores) / len(valid_scores)
    #clarity_score = sum(clarity_score) / len(clarity_score)
    return {'Relevance': rel_score, 'Correctness': corr_score, 'Coverage': cover_score,'Clarity': clarity_score}

# import pandas as pd
# df1 = pd.read_csv('output_analysis/gemini-2.5-flash_few.csv', encoding='utf-8')
# df2 = pd.read_csv('output_analysis/gemini-2.5-flash_zero.csv', encoding='utf-8')
#
# # 转成集合
# ids1 = set(df1["id"])
# ids2 = set(df2["id"])
#
# # 找缺失的
# missing_in_file2 = ids1 - ids2
# missing_in_file1 = ids2 - ids1
# print(list(ids2).index(3106))
# print("file2 缺失的 id:", missing_in_file2)
# print("file1 缺失的 id:", missing_in_file1)
with open('Final_data_classification.json', encoding='utf-8') as f:
    data = json.load(f)
f.close()
classification = []
for d in data:
    classification.append(d['classification'])

file_list = os.listdir('result_new')
save_dic_method = {}
save_dic_source = {}
for file_name in file_list:
    print(file_name)
    method_result = []
    resource_result = []
    with open('result_new/'+file_name, encoding='utf-8') as f:
        result = json.load(f)
        f.close()
    with open('results_rebuttal/'+file_name, encoding='utf-8') as f:
        result_cla = json.load(f)
        f.close()

    for idx, cls in enumerate(classification):
        if file_name == 'gemini-2.5-flash_few.json' and idx == 992:
            continue
        elif file_name == 'gemini-2.5-flash_few.json' and idx > 992:
            idx = idx - 1

        if cls == 'Methodological':
            result[idx]['Model']['Clarity']=result_cla[idx]['Model']['Clarity']
            method_result.append(result[idx])
        elif cls == 'Resource':
            result[idx]['Model']['Clarity'] = result_cla[idx]['Model']['Clarity']
            resource_result.append(result[idx])
    performance_method = average_cal(method_result)
    performance_resource = average_cal(resource_result)
    key_llm = file_name[:-5]
    save_dic_method[key_llm] = performance_method
    save_dic_source[key_llm] = performance_resource
print()
with open('result_dict_method.json', 'w') as f:
    json.dump(save_dic_method, f)
    f.close()
with open('result_dict_resource.json', 'w') as f:
    json.dump(save_dic_source, f)
    f.close()
print(classification)
