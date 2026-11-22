# textcnn (word)

config: `default` | seed: 42 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 8

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7892 | 0.7894 | 0.7893 | 0.6833 | 0.0897 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7518    0.8188    0.7839      1258
     Neutral     0.7887    0.7397    0.7634      1095
    Positive     0.8380    0.8047    0.8210      1106

    accuracy                         0.7892      3459
   macro avg     0.7929    0.7877    0.7894      3459
weighted avg     0.7911    0.7892    0.7893      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7936 | 0.7244 | True |
| 2 | 0.5875 | 0.7576 | True |
| 3 | 0.4788 | 0.7728 | True |
| 4 | 0.3963 | 0.7756 | True |
| 5 | 0.3272 | 0.7814 | True |
| 6 | 0.2812 | 0.7856 | True |
| 7 | 0.2319 | 0.7847 | False |
| 8 | 0.2057 | 0.7899 | True |
| 9 | 0.1723 | 0.7839 | False |
| 10 | 0.1494 | 0.7838 | False |
