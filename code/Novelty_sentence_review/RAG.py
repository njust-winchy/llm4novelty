from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class NoveltyRetriever:
    def __init__(self, novelty_sentences, model_name='all-mpnet-base-v2'):
        self.novelty_sentences = novelty_sentences
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.index_sentences = None
        self._build_index()

    def _build_index(self):
        embeddings = self.model.encode(self.novelty_sentences, show_progress_bar=True, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
        self.index.add(embeddings)
        self.index_sentences = self.novelty_sentences  # Keep for lookup

    def retrieve(self, query_sentence, top_k=3):
        query_embedding = self.model.encode([query_sentence], convert_to_numpy=True)
        D, I = self.index.search(query_embedding, top_k)
        return [self.index_sentences[i] for i in I[0]]

    def batch_retrieve(self, query_sentences, top_k=3):
        query_embeddings = self.model.encode(query_sentences, convert_to_numpy=True)
        D, I = self.index.search(query_embeddings, top_k)
        results = []
        for indices in I:
            results.append([self.index_sentences[i] for i in indices])
        return results



