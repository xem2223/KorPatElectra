import os, sys, subprocess, textwrap, json, warnings
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import pandas as pd

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_TORCHVISION_IMPORTS"] = "1"
warnings.filterwarnings("ignore")

# 2. CSV 경로·HF 토큰·파라미터
CSV_IN  = "특허.csv" 

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
df = pd.read_csv(CSV_IN)
TEXT_COLUMN = 'abstract' if 'abstract' in df.columns else df.columns[0]
texts = df[TEXT_COLUMN].fillna("").tolist()

# -------------- LSA 임베딩 생성 --------------
print("Generating LSA embeddings...")
tfidf       = TfidfVectorizer(max_df=0.8, min_df=5, ngram_range=(1,2))
X_tfidf     = tfidf.fit_transform(texts)               # (n_docs × n_terms)
svd         = TruncatedSVD(n_components=768, random_state=42)
X_lsa_raw   = svd.fit_transform(X_tfidf)               # (n_docs × 768)
normalizer  = Normalizer(copy=False)
X_lsa       = normalizer.fit_transform(X_lsa_raw)      # 코사인 유사도용 정규화

# -------------- UMAP 차원 축소 --------------
print("Reducing dimensions with UMAP for LSA...")
reducer_lsa = umap.UMAP(
    n_neighbors=UMAP_NEIGHBORS,
    n_components=UMAP_COMPONENTS,
    metric=UMAP_METRIC,
    random_state=42
)
umap_lsa_embeddings = reducer_lsa.fit_transform(X_lsa)

# -------------- HDBSCAN 클러스터링 --------------
print("Clustering with HDBSCAN on LSA embeddings...")
clusterer_lsa = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=2,
    metric='euclidean',
    cluster_selection_method='leaf',
    cluster_selection_epsilon=0.15
)
cluster_labels_lsa = clusterer_lsa.fit_predict(umap_lsa_embeddings)
df['cluster_lsa'] = cluster_labels_lsa

# -------------- 노이즈 비율 계산 --------------
total_points_lsa = len(cluster_labels_lsa)
noise_count_lsa = (cluster_labels_lsa == -1).sum()
noise_ratio_lsa = noise_count_lsa / total_points_lsa * 100
print(f"LSA Noise points: {noise_count_lsa} / {total_points_lsa} ({noise_ratio_lsa:.2f}%)")


# -------------- LSA 클러스터링 품질 평가 --------------
print("\nEvaluating clustering quality on LSA UMAP embeddings (noise excluded)...")
mask_lsa = cluster_labels_lsa != -1
if mask_lsa.sum() > 1:
    sil_lsa = silhouette_score(
        umap_lsa_embeddings[mask_lsa],
        cluster_labels_lsa[mask_lsa]
    )
    db_lsa = davies_bouldin_score(
        umap_lsa_embeddings[mask_lsa],
        cluster_labels_lsa[mask_lsa]
    )
    ch_lsa = calinski_harabasz_score(
        umap_lsa_embeddings[mask_lsa],
        cluster_labels_lsa[mask_lsa]
    )
    print(f"Silhouette Score:    {sil_lsa:.4f}")
    print(f"Davies-Bouldin Index: {db_lsa:.4f}")
    print(f"Calinski-Harabasz:    {ch_lsa:.4f}")
else:
    print("충분한 군집화 포인트가 없어 품질 평가를 수행할 수 없습니다.")

CSV_OUT="클러스터링_LSA.csv"
df.to_csv(CSV_OUT, index=False)
print(f"Clustered data saved to: {CSV_OUT}")
