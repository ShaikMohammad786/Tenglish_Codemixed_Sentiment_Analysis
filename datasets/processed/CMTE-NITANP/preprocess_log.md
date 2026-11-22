# Preprocessing log - CMTE-NITANP

| step | rows | delta |
|---|---:|---:|
| parsed | 34,588 |  |
| after cleaning | 34,588 | +0 |
| drop empty text_clean | 34,588 | +0 |
| drop invalid/missing label | 34,588 | +0 |

## Signal audit (rows containing, before cleaning)

| signal | rows |
|---|---:|
| has_!?_punct | 0 |
| has_digit | 4,972 |
| has_emoji | 0 |
| has_hashtag | 0 |
| has_mention | 0 |
| has_url | 0 |

Emojis, punctuation, numbers and elongations are kept in `text_clean`.

## Length stats (text_clean: min / median / max)

| rows | tokens | chars |
|---:|---|---|
| 34,588 | 1 / 8 / 433 | 1 / 47 / 2731 |

## Labels

| label | rows |
|---|---:|
| Negative | 12,583 |
| Neutral | 10,944 |
| Positive | 11,061 |

## Normalization (Phase E)

- PLD dictionary entries: 9,388
- PLD applied to this dataset: yes
- Rows changed by normalization: 25,498 / 34,588
- Monolingual rows detected: 0
- Monolingual rows dropped: no (kept by default)

De-duplication and splitting are intentionally NOT done here.
