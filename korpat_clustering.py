import os, sys, subprocess, textwrap, json, warnings
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import collections

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION_IMPORTS"] = "1"
warnings.filterwarnings("ignore")

# 2. CSV 경로·HF 토큰·파라미터
CSV_IN  = "특허.csv"   # ← Colab 드라이브 경로면 수정
HF_TOKEN = "--"         # ← 본인 허깅페이스 토큰
MODEL_ID = "KIPI-ai/KorPatElectra"

# UMAP
UMAP_NEIGHBORS  = 40
UMAP_COMPONENTS = 2
UMAP_METRIC     = 'cosine'

# HDBSCAN
MIN_CLUSTER_SIZE       = 5
MIN_SAMPLES            = 2
CLUSTER_SELECTION_METHOD = 'leaf'

# 임베딩 배치 사이즈
BATCH_SIZE = 128
# -------------- 데이터 로드 및 전처리 --------------
df = pd.read_csv(CSV_IN)
TEXT_COLUMN = '요약' if '요약' in df.columns else df.columns[0]
texts = df[TEXT_COLUMN].fillna("").tolist()

# -------------- KorPatElectra 모델 로드 --------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_ID, use_auth_token=HF_TOKEN).to(device)
X_kpe  = model.encode(texts, batch_size=128, show_progress_bar=True, device=device)


# -------------- 텍스트 임베딩 --------------
print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    device=device
)

# -------------- UMAP 차원 축소 --------------
print("Reducing dimensions with UMAP...")
reducer = umap.UMAP(
    n_neighbors=UMAP_NEIGHBORS,
    n_components=UMAP_COMPONENTS,
    metric=UMAP_METRIC,
    random_state=42
)
umap_embeddings = reducer.fit_transform(embeddings)

# -------------- HDBSCAN 클러스터링 --------------
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=2,
    metric='euclidean',
    cluster_selection_method='leaf',
    cluster_selection_epsilon=0.1
)
cluster_labels = clusterer.fit_predict(umap_embeddings)
df['cluster'] = cluster_labels

# -------------- 노이즈 비율 계산 --------------
total_points = len(cluster_labels)
noise_count = (cluster_labels == -1).sum()
noise_ratio = noise_count / total_points * 100
print(f"Noise points: {noise_count} / {total_points} ({noise_ratio:.2f}%)")

# -------------- 클러스터 품질 평가 --------------
print("\nEvaluating clustering quality on UMAP embeddings (noise excluded)...")
mask = cluster_labels != -1
if mask.sum() > 1:
    sil_val = silhouette_score(umap_embeddings[mask], cluster_labels[mask])
    db_val  = davies_bouldin_score(umap_embeddings[mask], cluster_labels[mask])
    ch_val  = calinski_harabasz_score(umap_embeddings[mask], cluster_labels[mask])
    print(f"Silhouette Score: {sil_val:.4f}")
    print(f"Davies-Bouldin Index: {db_val:.4f}")
    print(f"Calinski-Harabasz Index: {ch_val:.4f}")
else:
    print("충분한 군집화 포인트가 없어 품질 평가를 수행할 수 없습니다.")

# -------------- 결과 저장 --------------
CSV_OUT="클러스터링_KPE.csv"
df.to_csv(CSV_OUT, index=False)
print(f"Clustered data saved to: {CSV_OUT}")
