# textcnn (word)

config: `default` | seed: 43 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 6

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7950 | 0.7950 | 0.7949 | 0.6920 | 0.0644 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.7642    0.8219    0.7920      1258
     Neutral     0.8012    0.7470    0.7732      1095
    Positive     0.8276    0.8119    0.8197      1106

    accuracy                         0.7950      3459
   macro avg     0.7977    0.7936    0.7950      3459
weighted avg     0.7962    0.7950    0.7949      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.8023 | 0.7382 | True |
| 2 | 0.5857 | 0.7663 | True |
| 3 | 0.4749 | 0.7785 | True |
| 4 | 0.3956 | 0.7773 | False |
| 5 | 0.3299 | 0.7806 | True |
| 6 | 0.2764 | 0.7915 | True |
| 7 | 0.2324 | 0.7885 | False |
| 8 | 0.1994 | 0.7860 | False |
