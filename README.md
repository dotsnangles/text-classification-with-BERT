# 뉴스 토픽 분류 AI 경진대회 벤치마킹 프로젝트 보고서

# 프로젝트 선정 이유

상위 입상자들의 코드를 벤치마킹하여 경쟁력 있는 Text Classification Model을 구축

공모전/대회의 절차와 방식을 체득

# 프로젝트 목표

Text Data Augmentation + Over Sampling

K-Fold Cross Validation

Logit Ensemble

# Libraries and Frameworks

Pandas, Matplotlib, Seaborn, Hanja, Translators, TextAttack Imblearn,

Huggingface APIs, PyTorch, etc.

# Text Data Augmentation + Over Sampling Result

45654 samples ⇒ 101864 samples

Back Translation 기법을 활용해 데이터를 증식하여 더 많은 Sample로 학습이 가능하게 했음

클래스별 샘플 수를 동일하게 맞춰 학습 시 클래스 편향을 방지하고자 했음

M2M-100 모델을 사용

Augmentation 후 신문기사 제목과 동일한 양식으로 후처리 진행

오류 데이터는 패턴을 찾아 삭제 혹은 수정

# Text Data Augmentation?

### Easy Data Augmentation

동의어/유의어 사전이 필요 ⇒ Synonym Replacement (SR) / Random Insertion (RI)

Sample만 가지고도 가능     ⇒ Random Swap (RS) / Random Deletion (RD)

⇒ 언어적 특성 그리고 사전의 한계로 SR과 RI까지 제대로 구현된 한국어 대상의 Library는 없는 듯함 

### Back Translation

원본을 타 언어로 번역한 뒤 다시 원본의 언어로 재번역하는 과정을 거쳐 증식하는 방법

EDA 기법이 문장 혹은 문단 단위로 적용된다고 생각하면 쉽다

⇒ 웹 기반의 번역기나 사전 학습된 트랜스포머 모델을 통해 가능

### Using Adversarial Examples

인간에게는 동일/유사하지만 모델에게는 전혀 다른 카테고리로 분류되는 examples를

Adversarial Attack을 사용해 생성한 뒤 훈련에 활용하는 방법

⇒ TextAttack이라는 라이브러리가 있으며 허깅페이스 모델들을 타겟으로 사용 가능한 것으로 보임

⇒ 한국어 NLP에 적용 가능한지는 좀 더 조사가 필요함

### Using Pre-Trained Transformers

Back Translation / Random Insertion / Random Replacement / Text Generation(문장을 조금 더 길게)

전부 사전 학습된 모델들을 통해 가능

⇒ Huggingface 파이프라인을 통해 구현 가능

### TDA 유의사항 (캐글 마스터의 가르침)

특별한 경우가 아닌 한 맥락과 문법을 유지하는 것이 관건

증식된 데이터에 모델이 과적합 되는 것을 방지하기 위해 Validation에는 원본 데이터만 사용하는 것이 좋다.

K-Fold시에는 과적합 방지를 위해 원본과 증식된 샘플을 같은 폴드에 포함해주는 것이 좋다.

NLP에서의 증식은 언제나 모델 성능에 도움을 주는 것은 아니기 때문에 다양한 증식 방법을 적용해보고 결과를 확인해보는 것이 좋다.

# K-Fold Cross Validation

모든 sample을 훈련에 활용 가능 ⇒ 정확도 Up

모든 sample을 평가에 활용 가능 ⇒ 데이터 편향 Down

# Logit Ensemble

Voting / Bagging / Boosting 중 Soft Voting에 해당

klue/roberta-base / bert-base-multilingual-uncased / xlm-roberta-base를 

각각 5-Fold로 학습한 뒤 총 15개의 모델로 앙상블을 진행함

## Huggingface APIs & PyTorch

직접 구현을 통해 절차와 방식을 이해해보려는 방편으로 접근이 쉬운 Huggingface APIs를 사용

PyTorch를 사용해 GPU 메모리 관리 (코랩 터짐 방지)

# 결과 및 나아가며

Public Score 기준 1등과 약 1.5점 차이까지 좁히는 데 성공함

31등으로 상위 10퍼센트 안에 들어가는 데 성공

Back Translation 외에도 Bert 계열 모델을 통한 Synonym Replacement (SR) / Random Insertion (RI) 구현 방법을 찾았기 때문에 추후 Data Augmentation 시 적용해볼 예정

단순 복제를 통한 오버샘플링이 아닌 증식만으로 이루어진 오버샘플링이라면 더 좋은 결과를 기대할 수 있을 것

klue/bert-base 등 더 나은 모델을 선택한다면 더 좋은 결과를 기대할 수 있을 것

하이퍼 파라미터도 초기값에서 변화를 준 것이 없기 때문에 레퍼런스를 참고하여 다양한 테스트를 해본다면 효과를 볼 것으로 기대

PyTorch에 익숙해진다면 좀 더 효율적으로 학습을 진행할 수 있을 것

증식 데이터로 1차 파인튜닝한 뒤 원본 데이터로 2차 파인튜닝을 한다면 어떤 결과가 나올지 궁금

PyTorch 해커톤 하실 분?
