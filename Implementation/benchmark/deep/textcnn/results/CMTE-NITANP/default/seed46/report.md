# textcnn (word)

config: `default` | seed: 46 | dataset: CMTE-NITANP | split: B | vocab: 17,300 | best epoch: 8

| split | accuracy | macro-F1 | weighted-F1 | MCC | ECE |
|---|---:|---:|---:|---:|---:|
| test | 0.7892 | 0.7889 | 0.7889 | 0.6838 | 0.0917 |

## Per-class (test)

```
              precision    recall  f1-score   support

    Negative     0.8030    0.7711    0.7867      1258
     Neutral     0.7725    0.7598    0.7661      1095
    Positive     0.7905    0.8391    0.8140      1106

    accuracy                         0.7892      3459
   macro avg     0.7887    0.7900    0.7889      3459
weighted avg     0.7893    0.7892    0.7889      3459
```

## Epoch history

| epoch | train loss | val macro-F1 | best |
|---:|---:|---:|:---:|
| 1 | 0.7969 | 0.7288 | True |
| 2 | 0.5839 | 0.7646 | True |
| 3 | 0.4730 | 0.7723 | True |
| 4 | 0.3981 | 0.7709 | False |
| 5 | 0.3241 | 0.7869 | True |
| 6 | 0.2822 | 0.7807 | False |
| 7 | 0.2348 | 0.7884 | True |
| 8 | 0.1962 | 0.7899 | True |
| 9 | 0.1710 | 0.7836 | False |
| 10 | 0.1456 | 0.7875 | False |
