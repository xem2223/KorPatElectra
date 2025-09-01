# lsa_clustering.py


# korpat_clustering.py

# classification.py
" KorPatElectra 임베딩 기반 클러스터링 vs LSA 임베딩 기반 클러스터링에 대해 각각의 분류 가능성(classifiability)을 RandomForestClassifier를 통해 비교 분석"

잘 분리된 클러스터라면 → 높은 분류 정확도

불명확한 클러스터라면 → 낮은 분류 정확도

전체 워크플로우 요약
| 단계 | 설명                                           |
| -- | -------------------------------------------- |
| 1  | 클러스터링 결과 CSV 로드 (`cluster != -1` 필터링)        |
| 2  | 상위 N개 클러스터만 선택 (15개)                         |
| 3  | Label Encoding으로 정수 라벨화                      |
| 4  | 해당 클러스터링에 사용된 임베딩을 선택 (KorPatElectra vs LSA) |
| 5  | Train/Test 나누고 RandomForest 학습               |
| 6  | Confusion Matrix + Classification Report 출력  |

# trend_pred.py
"KorPatElectra 기반 임베딩 + UMAP + HDBSCAN 클러스터링을 통해, 유사한 해결 과제와 수단을 가진 특허들을 그룹화. 이후 KeyBERT를 활용하여 각 기술 영역별 핵심 키워드를 도출하였으며, 이를 통해 클러스터별 기술 트렌드 파악 및 전략 수립 가능"

전체 워크플로우 요약
| 단계        | 설명                    |
| --------- | --------------------- |
| 1. CSV 로드 | 요약 텍스트 불러오기           |
| 2. 임베딩    | KorPatElectra로 문장 임베딩 |
| 3. 차원 축소  | UMAP (3D)             |
| 4. 군집화    | HDBSCAN               |
| 5. 품질 평가  | silhouette, db, ch    |
| 6. 키워드 추출 | KeyBERT 기반            |
| 7. 결과 저장  | 클러스터링 CSV + 키워드 CSV   |
