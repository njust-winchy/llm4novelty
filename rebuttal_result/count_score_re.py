#  -*- coding: utf-8 -*-
# import json
# import statistics
# variance_list = []
# with open('coling_rev_fin.json') as f:
#     data = json.load(f)
# for i in data:
#     reviews = i['reviews']
#     if len(i['reviews']) <2:
#         continue
#     score_list = []
#     for r in reviews:
#         score = r['review_scores']
#         score_list.append(int(score['Originality']))
#     sample_variance = statistics.variance(score_list)
#     variance_list.append(sample_variance)
#
# print(sum(variance_list)/len(variance_list))

#
# import json
# import statistics
# import pandas as pd
# import ast
# variance_list = []
# save_list = []
# with open('Final_data.json', encoding='utf-8') as f:
#     data_check = json.load(f)
# f.close()
# id_list = []
# for d in data_check:
#     id_list.append(int(d['id']))
# df = pd.read_csv('emnlp_23.csv')
# id = df['paper_id'].tolist()
# data = df['reviews'].apply(ast.literal_eval).tolist()
# for idx, rev in enumerate(data):
#     score_list = []
#     save_dic = {}
#     if id[idx] not in id_list:
#         continue
#     for r in rev:
#         confidence_score = r['meta']['reviewer_confidence'][0]
#         score_list.append(int(confidence_score))
#     # sample_variance = statistics.variance(score_list)
#     # if sample_variance > 1.5:
#     #     print(id[idx])
#     # variance_list.append(sample_variance)
#     if max(score_list)-min(score_list) >=3:
#         max_score = max(score_list)
#         min_score = min(score_list)
#         max_index = score_list.index(max_score)
#         min_index = score_list.index(min_score)
#         review_high_text = rev[max_index]['report']['paper_topic_and_main_contributions'] + rev[max_index]['report']['reasons_to_accept'] + rev[max_index]['report']['reasons_to_reject']
#         review_low_text = rev[min_index]['report']['paper_topic_and_main_contributions'] + rev[min_index]['report']['reasons_to_accept'] + rev[min_index]['report']['reasons_to_reject']
#         save_dic['id'] = id[idx]
#         save_dic['review_high'] = review_high_text
#         save_dic['review_low'] = review_low_text
#         save_list.append(save_dic)
# with open('review_confidence.json', 'w', encoding='utf-8') as f:
#     json.dump(save_list, f, ensure_ascii=False, indent=4)
# f.close()

#0.5702751993560292

import os
import pandas as pd
from tqdm import tqdm
import json
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
file_list = os.listdir('output_analysis')
with open('review_confidence.json', encoding='utf-8') as outfile:
    json_data = json.load(outfile)
outfile.close()
model = SentenceTransformer("all-MiniLM-L6-v2")
save_list = []
for filename in file_list:
    high_list = []
    low_list = []
    df = pd.read_csv(os.path.join('output_analysis', filename))
    save_dic = {}
    print(filename)
    if 'DeepReviewer' in filename:
        k = 'fast_output'
    else:
        k='output'
    for d in tqdm(json_data):

        target_id = d['id']
        result = df[df["id"] == target_id]
        if not result.empty:
            llm_output = result.iloc[0][k]
        else:
            llm_output = ""

        high_text = d['review_high']
        low_text = d['review_low']
        llm_embs = model.encode(llm_output, convert_to_numpy=True)
        high_embs = model.encode(high_text, convert_to_numpy=True)
        low_embs = model.encode(low_text, convert_to_numpy=True)
        high_sim = cosine_similarity(llm_embs, high_embs)
        low_sim = cosine_similarity(llm_embs, low_embs)
        high_list.append(high_sim)
        low_list.append(low_sim)
    h_s = sum(high_list)/len(high_list)
    low_s = sum(low_list)/len(low_list)
    llm_name = filename[:-4]
    save_dic[llm_name] = {'high': h_s, 'low': low_s}
    save_list.append(save_dic)


with open('review_confidence_result.json', 'w', encoding='utf-8') as outfile:
    json.dump(save_list, outfile, ensure_ascii=False, indent=4)
outfile.close()