# bilstm (word)

config: `default` | seed: 43 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 5

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7956 | 0.7951 | 0.7954 | 0.6927 | 0.0554 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7802    0.8211    0.8002      1258
     Neutral     0.7841    0.7498    0.7666      1095
    Positive     0.8254    0.8119    0.8186      1106

    accuracy                         0.7956      3459
   macro avg     0.7966    0.7943    0.7951      3459
weighted avg     0.7959    0.7956    0.7954      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.8015 | 0.7455 | True |
| 2 | 0.5719 | 0.7756 | True |
| 3 | 0.4883 | 0.7848 | True |
| 4 | 0.4085 | 0.7939 | True |
| 5 | 0.3448 | 0.7986 | True |
| 6 | 0.2874 | 0.7930 | False |
| 7 | 0.2364 | 0.7977 | False |
