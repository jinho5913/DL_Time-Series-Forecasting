# 샐러드 수요량 예측 모델 개발

[ 결과보고서 ]
https://spotted-fridge-c67.notion.site/92d8463500d44870a4c557ae66e4b0fc

1. 시계열 데이터 생성
  * Sliding Window 사용
2. LSTM(1 layer)를 사용
  * 데이터가 부족했기 때문에 Model Capacity 고려

train : val : test = 400 : 100 : 100
___
```
# for train
python train.py
   -> input sid
```


Test result

![image](https://user-images.githubusercontent.com/87609200/220369538-a835ad5a-bc99-45f8-830b-52049c5ea766.png)


