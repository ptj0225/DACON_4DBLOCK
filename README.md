# DACON_4DBLOCK

DACON에서 진행한 컴퓨터 비전 경진대회 솔루션입니다.

[DACON 4D BLOCK 경진대회 Private 8위](https://dacon.io/competitions/official/236046/codeshare/7533?page=1&dtype=recent)

[배경] 
포디블록(4Dblock)은 유아기∼전 연령을 대상으로 하는 융합놀이를 위한 조작교구 입니다.

블록놀이는 공간 지각능력과 창의력 발달에 도움이 되는 놀이도구이며, 교육용 블록교구는 다양하게 구성 및 조적하게 되어있어 비구조적인 특징을 갖습니다.

하지만 이러한 비구조적인 특성은 발달특성상 유아들에게는 목적 없는 놀이도구로 소진되기 쉽기 때문에 보다 창의적인 높은 수준의 블록놀이를 촉진하기 위해서 체계적인 쌓기 구조에 대한 사전지식의 지원이 필요합니다.

이를 위해 어린이의 쌓기 구조 데이터를 수집하고 이에 대한 반복적인 블록 쌓기 구조 패턴 인식 및 쌓기 구조의 패턴을 분류하여 효율적이고 유용한 방법 및 해결책을 제시할 수 있을 것입니다.

이 기술은 나아가 오프라인 실험군, 통제군을 대상으로한 공간지각력, 창의성 등 자체 개발 된 평가 툴을 추가 학습시켜 사용자의 융합적 레벨 테스트를 같이 제공하여 블록 놀이&활동을 통한 교육적 진단 서비스로 확장 하고자 합니다.

학습자 선호 유형에 따른 활동 및 프로그램 매칭에 적용할 2D 이미지 기반 블록 구조 추출 AI 모델을 만들어 주세요.



[주제]
2D 이미지 기반 블록 구조 추출 AI 모델 개발



[설명]
본 경진대회에서는 2D 이미지 내 포디블록의 10가지의 블록 패턴들의 존재 여부를 분류하는 Multi-Label Classification을 수행해야합니다.

또한 실험 환경에서 촬영된 이미지가 학습 데이터로 주어지며, 평가(테스트 데이터)는 실제 환경에서 촬영된 이미지로 이루어집니다.


[주최 / 주관]
주최: 포디랜드, AI Frenz
주관: 데이콘


[참가자격]
일반인, 학생 등 누구나 대회 참가 가능


폴더 구조는 아래와 같습니다.

    ./indoorCVPR_09/
    ./train/
    ./test/
    ./train.csv
    ./test.csv
    ./sample_submission.csv
    ./util.ipynb
    ./main.ipynb

indoorCVPR_09의 경우 [indoor CVPR 09](https://web.mit.edu/torralba/www/indoor.html)에서 다운로드 받아주세요.

[DACON 4D BLOCK 경진대회 Private 8위](https://dacon.io/competitions/official/236046/codeshare/7533?page=1&dtype=recent)
에서 모델 설명 PPT 파일을 확인할 수 있습니다.