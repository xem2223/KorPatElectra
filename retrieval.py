import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors

# ------------------------
# 0) 설정
# ------------------------
CSV_IN   = os.getenv("CSV_IN", "OS_Matrics[ED].csv")
MODEL_ID = os.getenv("MODEL_ID", "KIPI-ai/KorPatElectra")
HF_TOKEN = os.getenv("HF_TOKEN")  # 반드시 환경변수로 설정할 것

assert HF_TOKEN is not None, "환경변수 HF_TOKEN을 설정하세요 (export HF_TOKEN=...)"

# ------------------------
# 1) 데이터 로드 (단일 컬럼 규격화)
# ------------------------
df = pd.read_csv(CSV_IN)
df = df.reset_index(drop=True)
TEXT_COLUMN = '요약' if '요약' in df.columns else df.columns[0]
texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

# ------------------------
# 2) KorPatElectra 임베딩
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_ID, use_auth_token=HF_TOKEN).to(device)
X_kpe = model.encode(texts, batch_size=128, show_progress_bar=True, device=device)

# ------------------------
# 3) LSA 임베딩 (TF-IDF → SVD → Normalizer, 쿼리에도 동일 변환 적용)
# ------------------------
tfidf = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(texts)

n_terms = X_tfidf.shape[1]
n_comp = min(768, n_terms - 1) if n_terms > 1 else 1
svd = TruncatedSVD(n_components=n_comp, random_state=42)
X_lsa_raw = svd.fit_transform(X_tfidf)
normalizer = Normalizer(copy=False)
X_lsa = normalizer.fit_transform(X_lsa_raw)  # 문서 정규화

# ------------------------
# 4) 쿼리와 정답 세트
# ------------------------
query_texts = [
    "딥러닝을 이용한 고해상도 이미지 생성 방법",     # 이미지 생성
    "텍스트 생성을 위한 언어 모델을 생성하기 위한 방법"  # 텍스트 생성
]
ground_truth = {
    0: [27, 76, 744, 880, 871, 1076, 1890, 1692, 383, 991, 1994],
    1: [489, 1621, 1972, 744, 763, 888, 1074, 1751, 1798],
}
# 유효 인덱스 필터 (ground truth가 df 길이 초과하는 경우 대비)
N = len(df)
ground_truth = {qi: [idx for idx in idxs if 0 <= idx < N] for qi, idxs in ground_truth.items()}

# ------------------------
# 5) 쿼리 임베딩 (KPE/LSA 모두 코퍼스 파이프라인과 동일 처리)
# ------------------------
# KPE
query_emb_kpe = model.encode(query_texts, batch_size=8, device=device)
query_emb_kpe = l2norm(query_emb_kpe)

# LSA
query_tfidf = tfidf.transform(query_texts)      # fit() 금지, 반드시 transform()
query_lsa_raw = svd.transform(query_tfidf)      # 코퍼스에 맞춰 학습된 SVD 사용
query_emb_lsa = normalizer.transform(query_lsa_raw)

# ------------------------
# 6) 평가 함수 (NearestNeighbors로 Top-k만)
# ------------------------
def evaluate_queries_knn(query_emb, doc_emb, ground_truth, model_name="Model", k=10):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(doc_emb)), metric='cosine')
    nbrs.fit(doc_emb)

    Pk_list, Rk_list, MRRk_list = [], [], []
    per_query = []

    for qi in range(len(query_emb)):
        true_set = set(ground_truth.get(qi, []))
        if not true_set:
            continue

        distances, indices = nbrs.kneighbors(query_emb[qi:qi+1], return_distance=True)
        topk_idx = indices[0].tolist()

        hits = [1 if idx in true_set else 0 for idx in topk_idx]
        Pk = sum(hits) / len(topk_idx)
        Rk = sum(hits) / max(1, len(true_set))

        # MRR@k
        mrr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h == 1:
                mrr = 1.0 / rank
                break

        Pk_list.append(Pk)
        Rk_list.append(Rk)
        MRRk_list.append(mrr)
        per_query.append((qi, Pk, Rk, mrr, topk_idx))

    print(f"\n📌 [{model_name}] Query-based Evaluation (Top-{len(topk_idx)})")
    print(f"Precision@k: {np.mean(Pk_list):.4f}")
    print(f"Recall@k:    {np.mean(Rk_list):.4f}")
    print(f"MRR@k:       {np.mean(MRRk_list):.4f}")

    # 각 쿼리별 상세
    for qi, Pk, Rk, mrr, idxs in per_query:
        print(f"  - Q{qi}: P@k={Pk:.3f}, R@k={Rk:.3f}, MRR@k={mrr:.3f}, topk_head={idxs[:5]}")

# ------------------------
# 7) 실행
# ------------------------
evaluate_queries_knn(query_emb_kpe, X_kpe, ground_truth, model_name="KorPatElectra", k=10)
evaluate_queries_knn(query_emb_lsa, X_lsa, ground_truth, model_name="LSA",           k=10)
