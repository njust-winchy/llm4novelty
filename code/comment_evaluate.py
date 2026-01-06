# -*- coding: utf-8 -*-
import json

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import numpy as np
import ast
from collections import Counter
import math
import re
from scientific_information_change.estimate_similarity import SimilarityEstimator


estimator = SimilarityEstimator()
def evaluate_distribution_and_kl(y_true, y_pred, labels=None):
    """

      - distribution_accuracy

    """

    # 自动获取标签集合
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    true_cnt = Counter(y_true)
    pred_cnt = Counter(y_pred)

    total_true = sum(true_cnt.values())
    total_pred = sum(pred_cnt.values())

    # 1. 类别分布
    distribution_true = {lab: true_cnt[lab] / total_true for lab in labels}
    distribution_pred = {lab: pred_cnt[lab] / total_pred for lab in labels}

    # 2. 分布相似度 (1 - L1_norm / 2)
    l1 = sum(abs(distribution_true[lab] - distribution_pred[lab]) for lab in labels)
    distribution_accuracy = 1 - l1 / 2

    return {
        "distribution_accuracy": distribution_accuracy,
    }
def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
def evaluate_single_set(intro_sentences, evals, gold_evals=None, model=None):
    """
    ��ĳһ������ (ר�� �� ģ��) ���Զ�����
    intro_sentences: ������ӱ�Ծ���
    evals: list of (label, text)  �� list of str (û�б�ǩʱ)
    gold_evals: gold ��ע (ר������), ���� correctness �Ա�
    """
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

    # ���������Ƿ����ǩ
    if isinstance(evals[0], tuple):
        labels, texts = zip(*evals)
    else:
        labels, texts = [None] * len(evals), evals

    # ---- 1. Relevance ----
    intro_emb = model.encode(" ".join(intro_sentences), convert_to_numpy=True)
    eval_embs = model.encode(texts, convert_to_numpy=True)
    relevance = estimator.estimate_ims(intro_sentences, texts)
    relevance = relevance.max(axis=1)
    relevance = relevance.mean()
    # ---- 2. Correctness ----
    if gold_evals is None:  #
        gold_labels = [lbl for lbl, _ in evals]
        correctness = evaluate_distribution_and_kl(gold_labels, gold_labels)
    else:
        gold_labels = [lbl for lbl, _ in gold_evals]
        pred_labels = [t for t in labels]
        correctness = evaluate_distribution_and_kl(gold_labels, pred_labels)


    # ---- 3. Coverage ----
    if gold_evals is None:  # ר���Լ����Լ� = 100%
        coverage = 1.0
    else:
        gold_texts = [txt for _, txt in gold_evals]
        gold_embs = model.encode(gold_texts, convert_to_numpy=True)
        coverage_hits = 0
        for g in gold_embs:
            sims = [cosine_similarity(g, m) for m in eval_embs]
            if max(sims) >= 0.7:
                coverage_hits += 1
        coverage = coverage_hits / len(gold_embs) if len(gold_embs) > 0 else 0.0

    # ---- 4. Clarity ----
    keywords = [w for s in intro_sentences for w in re.findall(r"\b[A-Za-z0-9\-]+\b", s) if len(w) > 5]
    #print(type(keywords), keywords)

    keyword_count = sum(any(k.lower() in t.lower() for k in keywords) for t in texts)
    keyword_ratio = keyword_count / len(texts) if texts else 0.0
    avg_len = np.mean([len(t.split()) for t in texts]) if texts else 0
    clarity = (0.5 * keyword_ratio) + (0.5 * min(avg_len / 20, 1))

    return {
        "Relevance": float(relevance),
        "Correctness": correctness,
        "Coverage": coverage,
        "Clarity": clarity
    }

model = SentenceTransformer("all-MiniLM-L6-v2")

# takes a list of sentences A of length N and a list of sentences B of length M and returns a numpy array S of size N×M,
# where S_{ij} is the IMS between A_i and B_j.
def load_dict(x):
    try:
        return ast.literal_eval(x)
    except:
        return None  # 解析失败时返回 None
def transfer(data):
    expert = []

    for label, items in data.items():
        for content in items:
            if not isinstance(content, str):
                continue
            c = content.strip()
            # 过滤 None、"None" 和长度 < 10 的内容
            if c.lower() == "none" or len(c) < 10:
                continue
            expert.append((label, c))
    return expert





result_list = os.listdir('output_analysis')
for filename in result_list:
    save_name = 'results/' + filename[:-4] + '.json'
    if os.path.exists(save_name):
        print(save_name+'saved')
        continue
    df = pd.read_csv('output_analysis'+'/'+filename)
    id = list(df['id'])
    llm_output = list(df['llm_output'].apply(load_dict))
    gold_output = list(df['gold_output'].apply(load_dict))
    introduction = list(df['paper_novelty'].apply(ast.literal_eval))
    run_index = len(introduction)
    result = []
    for i in tqdm(range(run_index)):
        sample_id = id[i]
        intro = introduction[i]
        model_evals = llm_output[i]
        expert_evals = gold_output[i]
        model_output = transfer(model_evals)
        expert = transfer(expert_evals)
        expert_result = evaluate_single_set(intro, expert, gold_evals=None, model=model)
        if len(model_output) == 0:
            model_result = {'Relevance': 0, 'Correctness': {'distribution_accuracy': 0}, 'Coverage': 0.0, 'Clarity': 0}
        else:
            model_result = evaluate_single_set(intro, model_output, gold_evals=expert, model=model)
        result.append({'Expert': expert_result, 'Model': model_result})
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    f.close()





# save_dic = {}
# result_list = os.listdir('result_new')
# for r_file in result_list:
#     with open(os.path.join('result_new', r_file), 'r') as f:
#         result = json.load(f)
#     f.close()
#     data = []
#     for r in tqdm(result):
#
#         row = {
#             "exp_relevance": r["Expert"]["Relevance"],
#             "model_relevance": r["Model"]["Relevance"],
#             "exp_coverage": r["Expert"]["Coverage"],
#             "model_coverage": r["Model"]["Coverage"],
#             "exp_clarity": r["Expert"]["Clarity"],
#             "model_clarity": r["Model"]["Clarity"],
#             "exp_dist_acc": r["Expert"]["Correctness"]["distribution_accuracy"],
#             "model_dist_acc": r["Model"]["Correctness"]["distribution_accuracy"],
#             "exp_kl": r["Expert"]["Correctness"]["kl_divergence"],
#             "model_kl": r["Model"]["Correctness"]["kl_divergence"]
#         }
#         data.append(row)
#
#     df = pd.DataFrame(data)
#     # 1. 描述统计
#     desc = df.describe()
#     mean_dict = desc.loc["mean"].to_dict()
#     key_llm = r_file[:-5]
#     save_dic[key_llm] = mean_dict
# with open('result_dict.json', 'w') as f:
#     json.dump(save_dic, f)
#     f.close()


