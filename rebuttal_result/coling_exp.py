import pandas as pd
import ast
from tqdm import tqdm
import json


example = []
df = pd.read_csv('coling_output-format.csv', encoding='utf-8')
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
    # with open(f'retrieve/{id}.json', encoding='utf-8') as f:
    #     retrieve_data = json.load(f)
    # f.close()
    n_r = output_format[idx]
    save_dic['id'] = str(id)
    save_dic['title'] = title
    save_dic['abstract'] = abstract
    save_dic['decision'] = decision
    save_dic['paper_novelty'] = n_p
    save_dic['review_novelty'] = r
    save_dic['output_format'] = n_r
    #save_dic['retrieved_title'] = retrieve_data[0]['title']
    #save_dic['retrieved_abstract'] = retrieve_data[0]['abstract']
    save_list.append(save_dic)

with open('Final_data_coling.json', 'w', encoding='utf-8') as f:
    json.dump(save_list, f, ensure_ascii=False, indent=4)