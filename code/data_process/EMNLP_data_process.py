import os
import json
from tqdm import tqdm
import pandas as pd
emnlp_23 = 'F:\code\\UKP_data\\EMNLP23\\data\\'
emnlp_24 = 'F:\code\\UKP_data\\ARR-EMNLP-2024\\data\\'


# EMNLP 2023 process
file_list = os.listdir(emnlp_23)
w_list = []
for n in tqdm(file_list):
    single_dict = {}
    reviews = []

    meta_file = emnlp_23 + n + '\\' + 'v1\\meta.json'
    review_file = emnlp_23 + n + '\\' + 'v1\\reviews.json'
    paper_file = emnlp_23 + n + '\\' + 'v2\\paper.itg.json'
    if os.path.exists(paper_file):
        with open(paper_file) as f:
            paper_data = json.load(f)
        f.close()
    elif os.path.exists(paper_file.replace('v2', 'v1')):
        with open(paper_file.replace('v2', 'v1')) as f:
            paper_data = json.load(f)
        f.close()
    else:
        continue
    with open(review_file) as f:
        review_data = json.load(f)
    f.close()
    with open(meta_file) as f:
        meta_data = json.load(f)
    f.close()

    for d in review_data:
        reviews.append(d)
    find_statu = False
    start_node = 0
    end_node = 0
    for idx, node in enumerate(paper_data['nodes']):
        if node['ntype'] == 'heading' and node['content'].lower() == 'introduction':
            start_node = idx
            find_statu = True
            continue
        if find_statu and node['ntype'] == 'heading':
            end_node = idx
            break
    print(n)
    single_dict['paper_id'] = n
    single_dict['title'] = meta_data['title']
    single_dict['abstract'] = meta_data['abstract']
    single_dict['keywords'] = meta_data['keywords']
    if 'status' in meta_data:
        single_dict['decision'] = meta_data['status']
    elif 'decision' in meta_data:
        single_dict['decision'] = meta_data['decision']
    if end_node==0:
        continue
    single_dict['paper_introduction'] = paper_data['nodes'][start_node:end_node]
    single_dict['reviews'] = reviews
    w_list.append(single_dict)
df = pd.DataFrame(w_list)
df.to_csv('emnlp_23.csv', index=False, header=True)

# EMNLP 2024 process
# file_list = os.listdir(emnlp_24)
# w_list = []
# for n in tqdm(file_list):
#     single_dict = {}
#     reviews = []
#
#     meta_file = emnlp_24 + n + '/' + 'v1/meta.json'
#     review_file = emnlp_24 + n + '/' + 'v1/reviews.json'
#     paper_file = emnlp_24 + n + '/' + 'v1/paper.itg.json'
#     if os.path.exists(paper_file):
#         with open(paper_file, encoding='utf-8') as f:
#             paper_data = json.load(f)
#         f.close()
#     else:
#         print('no paper data')
#         continue
#     with open(review_file, encoding='utf-8') as f:
#         review_data = json.load(f)
#     f.close()
#     if len(review_data)==0:
#         print('no review data')
#         continue
#     with open(meta_file, encoding='utf-8') as f:
#         meta_data = json.load(f)
#     f.close()
#     length = len(review_data)
#     for d in range(length):
#         reviews.append(review_data[str(d)])
#     find_statu = False
#     start_node = 0
#     end_node = 0
#     for idx, node in enumerate(paper_data['nodes']):
#         if node['ntype'] == 'heading' and node['content'].lower() == 'introduction':
#             start_node = idx
#             find_statu = True
#             continue
#         if find_statu and node['ntype'] == 'heading':
#             end_node = idx
#             break
#     #print(n)
#     single_dict['paper_id'] = n
#     single_dict['title'] = meta_data['title']
#     single_dict['abstract'] = meta_data['abstract']
#     single_dict['accepted_at'] = meta_data['accepted_at']
#     if end_node==0:
#         print('end node')
#         continue
#     single_dict['paper_introduction'] = paper_data['nodes'][start_node:end_node]
#     single_dict['reviews'] = reviews
#     w_list.append(single_dict)
# df = pd.DataFrame(w_list)
# df.to_csv('emnlp_24.csv', index=False, header=True)