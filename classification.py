from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------
# KorPatElectra 모델을 통해 클러스터링 후 분류 성능 평가
# ------------------------
input_csv = '클러스터링_KPE.csv'
df = pd.read_csv(input_csv)

# -1 (노이즈) 제거
df_kpe = df[df['cluster'] != -1].copy()

# 상위 10개 클러스터 선택
top_clusters = [cid for cid, cnt in Counter(df_kpe['cluster']).most_common(15)]
df_kpe = df_kpe[df_kpe['cluster'].isin(top_clusters)].copy()

# 라벨 인코딩
le_kpe = LabelEncoder()
y_kpe = le_kpe.fit_transform(df_kpe['cluster'])

# 임베딩 선택
X_kpe_selected = X_kpe[df_kpe.index]

# 학습
X_train, X_test, y_train, y_test = train_test_split(X_kpe_selected, y_kpe, stratify=y_kpe, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

print("Korpatelectra 임베딩 → Korpatelectra 클러스터 분류 성능")
print(classification_report(y_test, clf.predict(X_test), target_names=le_kpe.classes_.astype(str)))

# === Korpatelectra 클러스터 ===
y_pred_kpe = clf.predict(X_test)
cm_kpe = confusion_matrix(y_test, y_pred_kpe)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_kpe, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_kpe.classes_, yticklabels=le_kpe.classes_)
plt.title("Confusion Matrix: Korpatelectra")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ------------------------
# LSA을 통해 클러스터링 후 분류 성능 평가
# ------------------------
input_csv = '클러스터링_LSA.csv'
df = pd.read_csv(input_csv)

# -1 제거
df_lsa = df[df['cluster_lsa'] != -1].copy()

# 상위 10개 클러스터 선택
top_clusters_lsa = [cid for cid, cnt in Counter(df_lsa['cluster_lsa']).most_common(15)]
df_lsa = df_lsa[df_lsa['cluster_lsa'].isin(top_clusters_lsa)].copy()

# 라벨 인코딩
le_lsa = LabelEncoder()
y_lsa = le_lsa.fit_transform(df_lsa['cluster_lsa'])

# 임베딩 선택
X_lsa_selected = X_lsa[df_lsa.index]

# 학습
X_train, X_test, y_train, y_test = train_test_split(X_lsa_selected, y_lsa, stratify=y_lsa, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

print("LSA 임베딩 → LSA 클러스터 분류 성능")
print(classification_report(y_test, clf.predict(X_test), target_names=le_lsa.classes_.astype(str)))

# === LSA 클러스터 ===
y_pred_lsa = clf.predict(X_test)
cm_lsa = confusion_matrix(y_test, y_pred_lsa)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_lsa, annot=True, fmt='d', cmap='Greens',
            xticklabels=le_lsa.classes_, yticklabels=le_lsa.classes_)
plt.title("Confusion Matrix: LSA 임베딩")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
