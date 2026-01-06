import os
import json
data_id = os.listdir('COLING')
file_prior = 'COLING/'
save_list = []
for id in data_id:
    save_dic = {}
    paper_file = file_prior + id + '/v1/paper.itg.json'
    review_file = file_prior + id + '/v1/reviews.json'
    meta_file = file_prior + id + '/v1/meta.json'
    with open(paper_file) as f:
        paper_content = json.load(f)
    f.close()
    with open(review_file) as f:
        review_content = json.load(f)
    f.close()
    with open(meta_file) as f:
        meta_content = json.load(f)
    f.close()
    review_list = []
    for rev in review_content:
        review_dic = {}
        full_text = rev['report']['main']
        scores = rev['scores']
        sentence_slice = []
        for span in rev['meta']['sentences']['main']:
            strat = span[0]
            end = span[1]
            sentence_slice.append(full_text[strat:end])
        review_dic['review_text'] = full_text
        review_dic['review_scores'] = scores
        review_dic['review_sentences'] = sentence_slice
        review_list.append(review_dic)
    Title = meta_content['title']
    Abstract = meta_content['abstract']
    meta_file_2 = file_prior + id + '/v2/meta.json'
    if os.path.exists(meta_file_2):
        Decision = 'Accept'
    else:
        Decision = 'Reject'
    #Track = meta_content['track']
    find_statu = False
    start_node = 0
    end_node = 0
    for idx, node in enumerate(paper_content['nodes']):
        if node['ntype'] == 'heading' and node['content'].lower() == 'introduction':
            start_node = idx
            find_statu = True
            continue
        if find_statu and node['ntype'] == 'heading':
            end_node = idx
            break
    Paper_Introduciton = paper_content['nodes'][start_node:end_node]
    save_dic['paper_id'] = id
    save_dic['title'] = Title
    save_dic['abstract'] = Abstract
    save_dic['decision'] = Decision
    #save_dic['track'] = Track
    save_dic['paper_introduction'] = Paper_Introduciton
    save_dic['reviews'] = review_list
    save_list.append(save_dic)


with open('Coling_2020.json', 'w') as f:
    json.dump(save_list, f)
f.close()

# print(len(save_list)) 89 Coling 2020