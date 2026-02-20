import json
from nltk import sent_tokenize
import pandas as pd
with open('coling_rev_fin.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
f.close()
save_list = []
for i in data:
    paper_introduction = i['paper_introduction']
    w_list = []
    review_list = []
    for d in paper_introduction:
        if 'introduction' in d['content'].lower() and d['ntype'] == 'heading':
            continue
        else:
            if len(d['content']) < 5:
                continue
            sentence_token = sent_tokenize(d['content'])

            for s in sentence_token:
                if len(s) < 10:
                    continue
                w_list.append(s)
    for w in i['reviews']:
        review_list+=w['review_sentences']
    i['review_sentence'] = review_list
    i['introduction_sentence'] = w_list
    save_list.append(i)
df = pd.DataFrame(save_list)
df.to_csv('coling_rebuttal.csv', index=False)