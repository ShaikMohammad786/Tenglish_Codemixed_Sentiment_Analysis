# Zero-shot transfer: CMTE-NITANP (Split B) → CMTET

CMTET rows: 19,868 | multi-cluster: 402 | conflicting-label clusters: 199

| model | in-domain F1 | CMTET F1 | transfer drop | CMTET acc | singleton F1 | conflict F1 |
|---|---:|---:|---:|---:|---:|---:|
| transformers/indicbert_lora | 0.8134 | 0.6380 | +0.1754 | 0.6551 | 0.6388 | 0.3472 |
| transformers/telugu_bert_lora | 0.8149 | 0.6334 | +0.1815 | 0.6464 | 0.6337 | 0.3735 |
| transformers/muril_fullft | 0.8336 | 0.6330 | +0.2006 | 0.6464 | 0.6332 | 0.3608 |
| transformers/muril_lora | 0.8239 | 0.6274 | +0.1965 | 0.6418 | 0.6276 | 0.3693 |
| transformers/xlmr_lora | 0.8035 | 0.5982 | +0.2053 | 0.6144 | 0.5978 | 0.3407 |
| transformers/mbert_lora | 0.8064 | 0.5845 | +0.2219 | 0.6020 | 0.5841 | 0.3622 |
| transformers/distilmbert_lora | 0.8073 | 0.5503 | +0.2570 | 0.5550 | 0.5499 | 0.3757 |
