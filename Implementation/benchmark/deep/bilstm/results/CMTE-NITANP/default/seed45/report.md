# bilstm (word)

config: `default` | seed: 45 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 6

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7936 | 0.7939 | 0.7935 | 0.6897 | 0.0685 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7641    0.8084    0.7856      1258
     Neutral     0.8102    0.7562    0.7822      1095
    Positive     0.8137    0.8137    0.8137      1106

    accuracy                         0.7936      3459
   macro avg     0.7960    0.7928    0.7939      3459
weighted avg     0.7946    0.7936    0.7935      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7914 | 0.7383 | True |
| 2 | 0.5697 | 0.7693 | True |
| 3 | 0.4959 | 0.7876 | True |
| 4 | 0.4120 | 0.7896 | True |
| 5 | 0.3499 | 0.7905 | True |
| 6 | 0.2929 | 0.7952 | True |
| 7 | 0.2457 | 0.7884 | False |
| 8 | 0.1999 | 0.7880 | False |
