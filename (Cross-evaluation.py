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

#  LSA 임베딩으로 cluster (KPE 클러스터) 예측
X = X_lsa[df_kpe.index]
y = le_kpe.transform(df_kpe['cluster'])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))

#  KorPatElectra 임베딩으로 cluster (LSA 클러스터) 예측
X_kpe_selected = X_kpe[df_lsa.index]
y = le_lsa.transform(df_lsa['cluster_lsa'])

X_train, X_test, y_train, y_test = train_test_split(
    X_kpe_selected, y, stratify=y_lsa, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

print("KorPatElectra 임베딩 → LSA 클러스터 예측 성능")
print(classification_report(y_test, clf.predict(X_test), target_names=le_lsa.classes_.astype(str)))
