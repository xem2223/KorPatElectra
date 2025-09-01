# lsa_clustering.py


# korpat_clustering.py


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
