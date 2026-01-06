# -*- coding: utf-8 -*-

import pandas as pd
import os
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import faiss
import numpy as np
import json



class AbstractSimilaritySearch:
    def __init__(self, all_abstracts, model_name='all-MiniLM-L6-v2', index_path='faiss_index_23.bin', vec_path='vectors_23.npy'):
        self.all_abstracts = all_abstracts
        self.model_name = model_name
        self.index_path = index_path
        self.vec_path = vec_path
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.abstract_vectors = None

    def build_index(self):
        print("开始生成向量...")
        self.abstract_vectors = self.model.encode(self.all_abstracts, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(self.abstract_vectors)
        dim = self.abstract_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.abstract_vectors)
        print("索引构建完成，开始保存索引和向量...")
        faiss.write_index(self.index, self.index_path)
        np.save(self.vec_path, self.abstract_vectors)
        print(f"索引保存到 {self.index_path}，向量保存到 {self.vec_path}")

    def load_index(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.vec_path):
            raise FileNotFoundError("索引文件或向量文件不存在，请先调用 build_index() 构建索引。")
        print("加载索引和向量...")
        self.index = faiss.read_index(self.index_path)
        self.abstract_vectors = np.load(self.vec_path)
        print("加载完成。")

    def search(self, query_abstract, top_k=5):
        if self.index is None:
            self.load_index()
        query_vector = self.model.encode([query_abstract], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        return indices[0].tolist(), distances[0].tolist()
user = os.getenv("DB_USER", "root")
password = os.getenv("DB_PASSWORD", "970515")
host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "3306")
database = os.getenv("DB_NAME", "ACL")

engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8')# 填写自己所需的SQL语句，可以是复杂的查询语句
sql_query = """
SELECT DISTINCT *
FROM paper
WHERE year BETWEEN 2019 AND 2022
  AND CHAR_LENGTH(abstract) > 10
  AND venue IN ('ACL', 'EMNLP', 'NAACL');
"""
# 使用pandas的read_sql_query函数执行SQL语句，并存入DataFrame
df = pd.read_sql(sql_query, engine)
df = df.drop_duplicates(subset=['title'])

candidate_title = list(df['title'])
candidate_abstract = list(df['abstract'])
url = list(df['url'])
candidate_venue = list(df['venue'])

searcher = AbstractSimilaritySearch(candidate_abstract)
# 如果是第一次，先建索引（耗时较长）
if not os.path.exists(searcher.index_path) or not os.path.exists(searcher.vec_path):
        searcher.build_index()
df_emnlp = pd.read_csv('emnlp_23_sen.csv')
if not os.path.exists('retrieve'):
    os.mkdir('retrieve')
for row in tqdm(df_emnlp.iterrows(), total=len(df_emnlp)):
    save_list = []
    save_dic = {}
    row = row[1]

    abstract = row['abstract']
    title = row['title']
    paper_id = row['paper_id']
    if os.path.exists(f"retrieve/{paper_id}.json"):
        continue
    indices, scores = searcher.search(abstract)

    pdf_title = []
    pdf_abstract = []
    for i in indices:
        pdf_title.append(candidate_title[i])
        pdf_abstract.append(candidate_abstract[i])
    save_dic['title'] = pdf_title
    save_dic['abstract'] = pdf_abstract
    save_list.append(save_dic)
    with open(f'retrieve/{paper_id}.json', 'w') as f:
        json.dump(save_list, f)
    f.close()
