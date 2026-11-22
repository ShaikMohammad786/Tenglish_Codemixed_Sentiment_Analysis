# Preprocessing log - CMTET

| step | rows | delta |
|---|---:|---:|
| parsed | 19,868 |  |
| after cleaning | 19,868 | +0 |
| drop empty text_clean | 19,868 | +0 |
| drop invalid/missing label | 19,868 | +0 |

## Signal audit (rows containing, before cleaning)

| signal | rows |
|---|---:|
| has_!?_punct | 1,257 |
| has_digit | 5,778 |
| has_emoji | 5,886 |
| has_hashtag | 1,370 |
| has_mention | 8,425 |
| has_url | 0 |

Emojis, punctuation, numbers and elongations are kept in `text_clean`.

## Length stats (text_clean: min / median / max)

| rows | tokens | chars |
|---:|---|---|
| 19,868 | 5 / 14 / 603 | 13 / 78 / 3409 |

## Labels

| label | rows |
|---|---:|
| Negative | 7,717 |
| Neutral | 4,222 |
| Positive | 7,929 |

## Normalization (Phase E)

- PLD dictionary entries: 9,388
- PLD applied to this dataset: no (dictionary is Telugu-specific)
- Rows changed by normalization: 0 / 19,868
- Monolingual rows detected: 3,599
- Monolingual rows dropped: no (kept by default)

De-duplication and splitting are intentionally NOT done here.
