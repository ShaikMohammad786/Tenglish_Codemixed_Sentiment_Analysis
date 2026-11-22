# textcnn (word)

config: `default` | seed: 45 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 10

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7973 | 0.7973 | 0.7974 | 0.6954 | 0.1018 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7727    0.8267    0.7988      1258
     Neutral     0.7887    0.7671    0.7778      1095
    Positive     0.8378    0.7939    0.8152      1106

    accuracy                         0.7973      3459
   macro avg     0.7997    0.7959    0.7973      3459
weighted avg     0.7986    0.7973    0.7974      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7885 | 0.7294 | True |
| 2 | 0.5784 | 0.7566 | True |
| 3 | 0.4738 | 0.7743 | True |
| 4 | 0.3964 | 0.7601 | False |
| 5 | 0.3316 | 0.7804 | True |
| 6 | 0.2754 | 0.7777 | False |
| 7 | 0.2342 | 0.7863 | True |
| 8 | 0.2058 | 0.7849 | False |
| 9 | 0.1727 | 0.7867 | True |
| 10 | 0.1521 | 0.7884 | True |
