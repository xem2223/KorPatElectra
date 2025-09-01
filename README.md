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

# Cross-evaluation.py
"두 가지 임베딩 (KorPatElectra vs LSA)을 비교하기 위해,
서로 다른 임베딩 공간에서 생성된 클러스터 라벨을 상대 임베딩으로 예측 가능한지 실험"

| 방향            | Accuracy | Macro F1 | 해석 요약                        |
| ------------- | -------- | -------- | ---------------------------- |
| **LSA → KPE** | 51%      | 0.33     | KPE 구조는 LSA에서 재현 어려움         |
| **KPE → LSA** | 61%      | 0.57     | LSA 구조는 KPE 공간에서 어느 정도 재현 가능 |

KorPatElectra 임베딩 → LSA 군집 예측은 정확도 0.61, F1-score 0.57로 전반적으로 높은 분류 성능을 보였다. 이는 KorPatElectra 임베딩이 LSA 기반 클러스터 구조를 보다 잘 설명할 수 있는 의미 정보를 내포하고 있음을 시사한다.

LSA 임베딩 → KorPatElectra 군집 예측은 F1-score가 0.33에 그쳤으며, 특히 recall이 현저히 낮은 결과를 보였다. 이는 LSA 임베딩이 KorPatElectra 기반 클러스터의 구조를 재현하기에 정보 손실이 크고, 의미 표현력이 제한적이라는 한계를 드러낸다.

결과적으로, KorPatElectra 임베딩은 LSA 군집 구조를 일정 부분 재현할 수 있지만, 반대의 경우는 어렵다는 점이 확인되었다. 이는 KorPatElectra가 더 풍부하고 정교한 의미 기반 표현을 학습하고 있음을 보여주는 근거가 된다.

# retrieval.py
"“쿼리 → 문서” 검색 성능을 KPE vs LSA로 비교·평가"
1. 두 가지 쿼리(“이미지 생성”, “텍스트 생성”)와 각 쿼리에 대응되는 정답 문서 인덱스 집합을 정의
2. 쿼리에 대해서도 각각 KorPatElectra 임베딩과 LSA 임베딩을 동일한 파이프라인으로 변환하여 생성
3. NearestNeighbors를 활용해 코사인 거리 기반 Top-k 유사 문서를 검색
4. 마지막으로 각 모델(KorPatElectra, LSA)에 대해 Precision@k, Recall@k, MRR@k를 계산하여 평균 성능과 쿼리별 검색 결과를 출력

실험 결과, KorPatElectra 기반 임베딩은 LSA 대비 정밀도에서 약 2.6배 우수한 성능을 보였으며, 재현율과 평균역순위(MRR) 또한 일관되게 상회하였다. 이는 KorPatElectra가 의미적 유사성을 보다 정교하게 포착하여, Top-k(여기서는 k=10) 결과 내에 실제 관련 문서를 높은 확률로 포함시키고, 관련 문서를 상위에 랭크하는 데 유리함을 시사한다.


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
