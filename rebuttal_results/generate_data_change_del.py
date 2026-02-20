# # -*- coding: utf-8 -*-
# import json
# import random
# import ast
# from openai import OpenAI
# from tqdm import tqdm
# prompt = '''Please paraphrase the following academic texts with these requirements:
# 1. Preserve the core academic meaning and technical details of the original texts
# 2. Make minor modifications using synonym replacement and sentence structure changes
# 3. Maintain the formality and professionalism of academic writing
# 4. Keep the output format as a list, consistent with the input format
# 5. Do not add new information not present in the original or delete important information
#
# Input texts:
# {input_texts}
#
# Please output only the paraphrased text list in the format: ['paraphrased text 1', 'paraphrased text 2', ...]
# Do not include any additional explanations.'''
#
#
# def remove_random_elements_inplace(data_list, num_to_remove):
#     """
#     从列表中随机移除指定数量的元素（原地修改）
#
#     参数:
#     data_list: 原始列表（会被修改）
#     num_to_remove: 要移除的元素数量
#
#     返回:
#     移除的元素列表
#     """
#     # 如果要移除的数量大于列表长度，调整数量
#     num_to_remove = min(num_to_remove, len(data_list))
#
#     if num_to_remove == 0:
#         return []
#
#     # 随机选择要移除的元素的索引（从大到小排序，避免索引变化问题）
#     indices_to_remove = sorted(random.sample(range(len(data_list)), num_to_remove), reverse=True)
#
#     # 记录被移除的元素
#     removed_elements = []
#
#     # 从后往前移除元素（避免索引变化）
#     for index in indices_to_remove:
#         removed_elements.append(data_list.pop(index))
#
#     return removed_elements
# def random_select_samples(data_list, num_samples=100):
#     """
#     从列表中随机选择指定数量的样本（不重复）
#
#     参数:
#     data_list: 原始列表
#     num_samples: 要选择的样本数量
#
#     返回:
#     随机选择的样本列表
#     """
#     # 如果要选择的样本数量大于列表长度，返回整个列表
#     if num_samples >= len(data_list):
#         print(f"警告：样本数量({num_samples})大于或等于列表长度({len(data_list)})，返回整个列表")
#         return data_list.copy()
#
#     # 随机选择不重复的样本
#     selected_samples = random.sample(data_list, num_samples)
#     return selected_samples
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
#         model='gpt-4o-mini'
#     )
#     return chat_completion.choices[0].message.content
# def replace_academic_texts(input_texts):
#     return f'''Please paraphrase the following academic texts with these requirements:
# 1. Preserve the core academic meaning and technical details of the original texts
# 2. Make minor modifications using synonym replacement and sentence structure changes
# 3. Maintain the formality and professionalism of academic writing
# 4. Keep the output format as a list, consistent with the input format
# 5. Do not add new information not present in the original or delete important information
# Input texts:
# \"\"\"
# {input_texts}
# \"\"\"
# Please output only the paraphrased text list in the format: ['paraphrased text 1', 'paraphrased text 2', ...]
# Do not include any additional explanations.'''
#
# with open('Final_data.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)
# f.close()
# save_list_1 = []
# save_list_2 = []
# result = random_select_samples(data, 100)
# for i in tqdm(result):
#
#     prompt = replace_academic_texts(i['paper_novelty'])
#     result_1 = gpt4task(prompt)
#     des_len = len(i['paper_novelty'])
#     rem = int(des_len * 0.8)
#     paper_novelty = i['paper_novelty']
#
#     if des_len == 1:
#         ex_1 = paper_novelty
#     else:
#         ex_1 = remove_random_elements_inplace(paper_novelty, rem)
#
#     # 创建两个不同版本的副本
#     dic_1 = i.copy()
#     dic_1['paper_novelty'] = ex_1
#     save_list_1.append(dic_1)
#
#     dic_2 = i.copy()
#     dic_2['paper_novelty'] = result_1
#     save_list_2.append(dic_2)
#
# with open('Final_data_del.json', 'w', encoding='utf-8') as f:
#     json.dump(save_list_1, f)
#     f.close()
# with open('Final_data_change.json', 'w', encoding='utf-8') as f:
#     json.dump(save_list_2, f)
#     f.close()


import json
with open('Final_data_del.json', encoding='utf-8') as f:
    data_del = json.load(f)
f.close()

with open('LLM_result/gpt-4o_rag.json', encoding='utf-8') as f:
    gpt4o_zero = json.load(f)
f.close()
save_list = []
for d in data_del:
    id = d['id']
    for g in gpt4o_zero:
        if g['id'] == id:
            save_list.append(g)
with open('change/gpt-4o_rag.json', 'w', encoding='utf-8') as f:
    json.dump(save_list, f)
f.close()
print()