import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import collections
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

# -------------- 데이터 로드 및 전처리 --------------
df = pd.read_csv(CSV_IN)
TEXT_COLUMN = '요약' if '요약' in df.columns else df.columns[0]
texts = df[TEXT_COLUMN].fillna("").tolist()

# -------------- KorPatElectra 모델 로드 --------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_ID, use_auth_token=HF_TOKEN).to(device)
X_kpe  = model.encode(texts, batch_size=128, show_progress_bar=True, device=device)

# -------------- LSA 임베딩 생성 --------------
print("Generating LSA embeddings...")
tfidf       = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,2))
X_tfidf     = tfidf.fit_transform(texts)               # (n_docs × n_terms)
svd         = TruncatedSVD(n_components=768, random_state=42)
X_lsa_raw   = svd.fit_transform(X_tfidf)               # (n_docs × 768)
normalizer  = Normalizer(copy=False)
X_lsa       = normalizer.fit_transform(X_lsa_raw)      # 코사인 유사도용 정규화

query = "텍스트 생성"
print(f"쿼리: {query}\n")

for i, text in enumerate(texts):
    if any(term in text for term in ["텍스트"]):  # 키워드 매칭 예시
        print(f"[{i}] {text[:100]}...")

# 사용자 쿼리 리스트
query_texts = [
    "딥러닝을 이용한 고해상도 이미지 생성 방법",      # 이미지 생성
    "텍스트 생성을 위한 언어 모델을 생성하기 위한 방법"              # 텍스트 생성
]

# 각 쿼리에 대응되는 정답 문서 인덱스 (사람이 직접 선정)
# 예: df['abstract']에서 의미적으로 가장 관련 있는 문서 index들
ground_truth = {
    0: [27, 76, 744, 880, 871, 1076, 1890, 1692, 383, 991, 1994 ],     # "이미지 생성" 쿼리에 대한 정답 인덱스들
    1: [489, 1621, 1972, 744,763, 888, 1074,1751, 1798]         # "텍스트 생성" 쿼리에 대한 정답 인덱스들
}

# KorPatElectra 임베딩
query_emb_kpe = model.encode(query_texts, device=device)

# LSA 임베딩
query_tfidf = tfidf.transform(query_texts)
query_emb_lsa_raw = svd.transform(query_tfidf)
query_emb_lsa = Normalizer(copy=False).fit_transform(query_emb_lsa_raw)

# 검색 및 평가 함수
def evaluate_queries(query_emb, doc_emb, ground_truth, model_name="Model", k=10):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    sims = cosine_similarity(query_emb, doc_emb)
    scores, recalls, mrrs = [], [], []

    for i, sim_row in enumerate(sims):
        true_set = set(ground_truth.get(i, []))
        if not true_set:
            continue

        topk_idx = np.argsort(sim_row)[::-1][:k]
        hits = [1 if idx in true_set else 0 for idx in topk_idx]

        precision = sum(hits) / k
        recall = sum(hits) / len(true_set)
        mrr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                mrr = 1.0 / rank
                break

        scores.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)

    print(f"\n📌 [{model_name}] Query-based Evaluation (Top-{k})")
    print(f"Precision@{k}: {np.mean(scores):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"MRR@{k}:       {np.mean(mrrs):.4f}")

evaluate_queries(query_emb_kpe, X_kpe, ground_truth, model_name="KorPatElectra")
evaluate_queries(query_emb_lsa, X_lsa, ground_truth, model_name="LSA")
