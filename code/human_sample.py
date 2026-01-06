# # -*- coding: utf-8 -*-
# import os
# import random
# import pandas as pd
#
# def load_model_outputs(folder="output_analysis"):
#     """
#     读取文件夹下所有模型的 CSV 结果。
#     返回:
#     models = {
#         "modelA": DataFrame,
#         "modelB": DataFrame,
#         ...
#     }
#     """
#     models = {}
#     for fname in os.listdir(folder):
#         if fname.endswith(".csv"):
#             model_name = os.path.splitext(fname)[0]
#             path = os.path.join(folder, fname)
#             df = pd.read_csv(path)
#             df = df.set_index("id")
#             models[model_name] = df
#     return models
#
#
# def sample_and_compare(models, n_samples=100):
#     """
#     从所有 sample_id 中随机选100个，
#     对每个 sample 随机选择两个模型进行对比。
#     返回 DataFrame
#     """
#     # 将所有模型的 sample_id 取交集（确保每个模型都有该 sample）
#     sample_sets = [set(df.index) for df in models.values()]
#     all_samples = set.intersection(*sample_sets)
#
#     if len(all_samples) < n_samples:
#         raise ValueError(f"可用 sample 数为 {len(all_samples)}，不足 {n_samples} 个可随机抽取")
#
#     selected_samples = random.sample(list(all_samples), n_samples)
#
#     model_names = list(models.keys())
#
#     records = []
#
#     for sid in selected_samples:
#         # 随机选择两个不同的模型
#         m1, m2 = random.sample(model_names, 2)
#
#         out1 = models[m1].loc[sid].to_dict()
#         out2 = models[m2].loc[sid].to_dict()
#
#         records.append({
#             "id": sid,
#             "model_1": m1,
#             **{f"model_1_{k}": v for k, v in out1.items()},
#             "model_2": m2,
#             **{f"model_2_{k}": v for k, v in out2.items()},
#         })
#
#     return pd.DataFrame(records)
#
#
# if __name__ == "__main__":
#     # 1. 加载所有模型 CSV
#     models = load_model_outputs("output_analysis")
#
#     # 2. 随机抽样 + 对比
#     df_compare = sample_and_compare(models, n_samples=100)
#
#     # 3. 保存结果到 CSV
#     df_compare.to_csv("random_model_comparisons.csv", index=False)
#
#     print("Saved as random_model_comparisons.csv")





import pandas as pd
import json

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import ast
from collections import Counter
import math
import re
from scientific_information_change.estimate_similarity import SimilarityEstimator


estimator = SimilarityEstimator()
def evaluate_distribution_and_kl(y_true, y_pred, labels=None):
    """
    只计算：
      - distribution_accuracy
      - KL divergence KL(T || P)
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
        "distribution_accuracy": distribution_accuracy
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

df = pd.read_csv('random_model_comparisons.csv')
id_list = list(df['id'])
model_1_output = df['model_1_output']
model_1_output_format = df['model_1_output_format']
model_1_llm_output = df['model_1_llm_output'].apply(load_dict)
model_1_gold_output = df['model_1_gold_output'].apply(load_dict)
model_1_paper_novelty = df['model_1_paper_novelty'].apply(ast.literal_eval)
model_2_output = df['model_2_output']
model_2_llm_output = df['model_2_llm_output'].apply(load_dict)
save_list = []
for i in tqdm(range(len(id_list))):
    save_dic = {}
    id = id_list[i]
    original_text_1 = model_1_output[i]
    model_output_1 = transfer(model_1_llm_output[i])
    expert = transfer(model_1_gold_output[i])
    model_output_2 = transfer(model_2_llm_output[i])
    original_text_2 = model_2_output[i]
    intro = model_1_paper_novelty[i]
    if len(model_output_1) == 0:
        expert_result_1 = {'Relevance': 0,
                        'Correctness': {'distribution_accuracy': 0, 'kl_divergence': 10.819778286704924},
                        'Coverage': 0.0, 'Clarity': 0}
        final_string_1 = 'Positive:\nNone\n\nNeutral:\nNone\n\nNegative:\nNone'
    else:
        expert_result_1 = evaluate_single_set(intro, model_output_1, gold_evals=expert, model=model)
        groups_1 = {"Positive": [], "Neutral": [], "Negative": []}
        for sentiment, text in model_output_1:
            groups_1[sentiment].append(text)

        # 拼接所有成一个字符串
        final_string_1 = (
                "Positive:\n" + "\n".join(groups_1["Positive"]) + "\n\n" +
                "Neutral:\n" + "\n".join(groups_1["Neutral"]) + "\n\n" +
                "Negative:\n" + "\n".join(groups_1["Negative"])
        )
    if len(model_output_2) == 0:
        expert_result_2 = {'Relevance': 0,
                        'Correctness': {'distribution_accuracy': 0},
                        'Coverage': 0.0, 'Clarity': 0}
        final_string_2 = 'Positive:\nNone\n\nNeutral:\nNone\n\nNegative:\nNone'
    else:
        expert_result_2 = evaluate_single_set(intro, model_output_2, gold_evals=expert, model=model)
        groups_2 = {"Positive": [], "Neutral": [], "Negative": []}
        for sentiment, text in model_output_2:
            groups_2[sentiment].append(text)

        # 拼接所有成一个字符串
        final_string_2 = (
                "Positive:\n" + "\n".join(groups_2["Positive"]) + "\n\n" +
                "Neutral:\n" + "\n".join(groups_2["Neutral"]) + "\n\n" +
                "Negative:\n" + "\n".join(groups_2["Negative"])
        )
    introduction = ''
    for j in intro:
        introduction = introduction + j +'\n'

    save_dic['id'] = id
    save_dic['review_text'] = model_1_output_format[i]
    save_dic['introduction'] = introduction
    save_dic['original_text_1'] = final_string_1
    save_dic['result_1'] = expert_result_1
    save_dic['original_text_2'] = final_string_2
    save_dic['result_2'] = expert_result_2
    save_list.append(save_dic)
pd.DataFrame(save_list).to_excel('random_human.xlsx', index=False)


import pandas as pd

df = pd.read_excel('random_human.xlsx')
df.pop('result_1')
df.pop('result_2')
df.to_excel('human_eval.xlsx', index=False)

