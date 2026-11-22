# textcnn (word)

config: `default` | seed: 44 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 8

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7936 | 0.7933 | 0.7934 | 0.6914 | 0.0887 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7371    0.8601    0.7938      1258
     Neutral     0.8105    0.7187    0.7619      1095
    Positive     0.8588    0.7920    0.8241      1106

    accuracy                         0.7936      3459
   macro avg     0.8021    0.7903    0.7933      3459
weighted avg     0.7992    0.7936    0.7934      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7940 | 0.7292 | True |
| 2 | 0.5844 | 0.7657 | True |
| 3 | 0.4804 | 0.7745 | True |
| 4 | 0.4011 | 0.7789 | True |
| 5 | 0.3405 | 0.7820 | True |
| 6 | 0.2801 | 0.7852 | True |
| 7 | 0.2355 | 0.7888 | True |
| 8 | 0.2055 | 0.7924 | True |
| 9 | 0.1738 | 0.7884 | False |
| 10 | 0.1570 | 0.7865 | False |
