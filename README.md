# trend_pred.py
| 단계        | 설명                    |
| --------- | --------------------- |
| 1. CSV 로드 | 요약 텍스트 불러오기           |
| 2. 임베딩    | KorPatElectra로 문장 임베딩 |
| 3. 차원 축소  | UMAP (3D)             |
| 4. 군집화    | HDBSCAN               |
| 5. 품질 평가  | silhouette, db, ch    |
| 6. 키워드 추출 | KeyBERT 기반            |
| 7. 결과 저장  | 클러스터링 CSV + 키워드 CSV   |
