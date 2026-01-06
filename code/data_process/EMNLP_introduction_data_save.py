import json
import pandas as pd
from nltk import sent_tokenize


# with open('conling_rev_fin.json', encoding='utf-8') as f:
#     data = json.load(f)
# f.close()
# w_list = []
# for d in data:
#     for l in d['paper_introduction']:
#         if 'introduction' in l['content'].lower() and l['ntype'] == 'heading':
#             continue
#         else:
#             if len(l['content'])<5:
#                 continue
#             sentence_token = sent_tokenize(l['content'])
#             for s in sentence_token:
#                 if len(s)<10:
#                     continue
#                 w_dict = {}
#                 w_dict['paper_id'] = d['paper_id']
#                 w_dict['sentence'] = s
#                 w_list.append(w_dict)
# print()
# df = pd.DataFrame(w_list)
# df.to_csv('labeling_data_zhao.csv', index=False, header=True, encoding='utf-8')

from nltk import sent_tokenize
import pandas as pd
import ast


df = pd.read_csv('emnlp_23.csv')
paper_introduction = df['paper_introduction'].apply(ast.literal_eval)
reviews = df['reviews'].apply(ast.literal_eval)
w_list = []
review_list = []
for d in paper_introduction:
    filter_sen = []
    for l in d:
        if 'introduction' in l['content'].lower() and l['ntype'] == 'heading':
            continue
        else:
            if len(l['content'])<5 or l['ntype'] != 'paragraph':
                continue
            sentence_token = sent_tokenize(l['content'])

            for s in sentence_token:
                if len(s)<10:
                    continue
                filter_sen.append(s)
    w_list.append(filter_sen)


# emnlp 2023
rev_list = []
for rev in reviews:
    process_list = []
    filter_rev = []

    for r in rev:
        process_list.append(r['report']['paper_topic_and_main_contributions'])
        process_list.append(r['report']['reasons_to_accept'])
        process_list.append(r['report']['reasons_to_reject'])
        for i in process_list:
            sen = sent_tokenize(i)
            for s in sen:
                if len(s)<10:
                    continue
                filter_rev.append(s)
    rev_list.append(filter_rev)
df['introduction_sentence'] = w_list
df['review_sentence'] = rev_list
df.to_csv('emnlp_23_sen.csv')

# emnlp 2024
# rev_list = []
# for rev in reviews:
#     process_list = []
#     filter_rev = []
#
#     for r in rev:
#         process_list.append(r['report']['paper_summary'])
#         process_list.append(r['report']['summary_of_strengths'])
#         process_list.append(r['report']['summary_of_weaknesses'])
#         for i in process_list:
#             sen = sent_tokenize(i)
#             for s in sen:
#                 if len(s)<10:
#                     continue
#                 filter_rev.append(s)
#     rev_list.append(filter_rev)
# df['introduction_sentence'] = w_list
# df['review_sentence'] = rev_list
# df.to_csv('emnlp_24_sen.csv')