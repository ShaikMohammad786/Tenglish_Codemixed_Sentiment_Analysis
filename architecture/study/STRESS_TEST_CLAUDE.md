# INDEPENDENT STRESS-TEST: OR-MuRIL PLAN (`FINAL_UNIFIED_RESEARCH_PLAN.md`)

**VERDICT: REFRAME** (Kill the "novel architecture / pretraining paradox" claims; reframe as a diagnostic audit and evaluation study).

---

### 1. Premise: The "Indic Pretraining Paradox" is Factually False
- **The Defect (§2.1, lines 44–48):** The plan claims MuRIL was trained "almost entirely on native Indic scripts" and cannot activate Indic subwords on Latin script. This is factually incorrect. Khanuja et al. (2021, §3) explicitly trained MuRIL on 17 Indic languages *and their transliterated counterparts* (via Dakshina and transliterated Wikipedia) using Translation Language Modeling (TLM). Its 197k vocabulary deliberately includes Romanized Indic subwords.
- **Consequence:** RQ1 (§2.2) collapses in its current phrasing. Reviewers familiar with Indic NLP will cite Khanuja et al. and reject the premise.
- **Salvageable Framing:** The real gap is not *script availability*, but *domain register*: MuRIL saw formal, rule-transliterated Wikipedia text, whereas social media Tenglish features unstandardized phonetic polygraphy, vowel elongations, and clitics.

### 2. Novelty: Method is Known; Venue Target is Fragile
- **Prior Art (§4.2):** Stochastic perturbation paired with symmetric Jensen-Shannon/KL consistency is mathematically equivalent to **UDA** (Xie et al., NeurIPS 2020) and **R-Drop** (Liang et al., NeurIPS 2021). Applying consistency loss over character perturbations to text classification is well-established.
- **Venue Viability:** At ACM TALLIP, claiming OR-MuRIL as a novel architectural contribution will face desk rejection. At ACL DravidianLangTech, it survives only as a shared-task system paper, not a breakthrough methodology.
- **Sharpening:** Do not claim an architectural invention. Frame the contribution as a controlled empirical study on whether consistency regularization provides genuine out-of-distribution robustness over simple data augmentation.

### 3. Feasibility / Compute: Glaring Internal Contradictions
- **The Contradiction (§5 vs §6 Table 1):** §5 promises "exactly TWO GPU training runs (~60 min total)". Yet §6 Table 1 specifies "Run across 3 seeds (42, 43, 44)". 3 seeds × 2 models = **6 training runs**, not 2.
- **Throughput Reality:** A dual-view pass ($x$ and $\tilde{x}$) requires two forward passes per batch step. On a T4 GPU (16 GB), MuRIL-base with LoRA on 27,670 rows takes ~18–22 min/epoch. A 3-epoch dual-view run takes ~60 min *per seed*.
- **Real Estimate:** 3 seeds of vanilla MuRIL (~60 min) + 3 seeds of OR-MuRIL (~180 min) = **~4.0 GPU hours**, not 60 minutes. If strictly held to 2 runs total, you must abandon multi-seed variance reporting.

### 4. Validity: Table 4 Stress-Test is Circular
- **Tautological Setup (§4.1 vs §6 Table 4):** §4.1 trains the model using synthetic transforms: $g \leftrightarrow v$, $ch \leftrightarrow ts$, and vowel elongation. Table 4 evaluates robustness using the exact same synthetic rules.
- **Consequence:** Demonstrating that a model trained on noise function $f(x)$ outperforms a baseline on noise function $f(x)$ is a closed-loop tautology. Reviewers will discard Table 4.
- **Fix:** Test on *held-out, disjoint* perturbations never seen in training: e.g., real human spelling variants mined from Twitter/YouTube, keyboard edit-distance noise, or natural slang substitutions.

### 5. Missing Controls & Confounded Transfer
- **Missing Baseline 1 (Plain Augmentation):** The plan lacks a MuRIL baseline trained on $x \cup \tilde{x}$ with standard Cross-Entropy. Without it, you cannot prove the $\mathcal{D}_\text{JS}$ consistency loss does anything beyond simple data augmentation.
- **Missing Baseline 2 (The 0-Parameter Hurdle):** TF-IDF + Logistic Regression already achieves **0.8293 macro-F1** on Split B. If OR-MuRIL achieves ~0.835, a 110M transformer barely ties a linear model.
- **CMTET Confound (§6 Table 3):** CMTET has a 55.4% label conflict rate on duplicate clusters and a 61-row spam template. A zero-shot drop on CMTET conflates domain shift with label noise.

### 6. The Bigger Picture: Terminal State Recommendation
- The project has pivoted 5 times (Leakage $\to$ SID-CM $\to$ MDS-Net $\to$ ByT5 $\to$ OR-MuRIL). Each pivot added fragile hypotheses that collapsed under scrutiny.
- Chasing marginal accuracy gains ($+0.01$) on a 27k training set is a losing strategy. 
- **The winning route:** Pivot to an **Empirical Audit and Robustness Benchmark**. When a linear model ($0.8293$) beats a deep CNN+BiLSTM ($0.7863$), the most impactful question is *evaluating the efficiency-robustness Pareto frontier*.

---

### Better RQs / What I Would Do Instead

1. **RQ1 (The Representation-Efficiency Audit):** Does parameter-heavy transformer pretraining (MuRIL, XLM-R) provide statistically significant benefits over n-gram linear models on Romanized Dravidian text under leakage-free evaluation?
2. **RQ2 (Consistency vs. Plain Augmentation):** Does representation consistency ($\mathcal{D}_\text{JS}$) outperform standard data augmentation when subjected to unseen, out-of-domain orthographic shifts?
3. **RQ3 (The Noise Sensitivity Cliff):** Under increasing rates of natural orthographic degradation, at what threshold do subword transformers degrade below character n-gram baselines?

**The Single Best Accuracy Lever:**  
**Ensembling TF-IDF Character N-Grams with MuRIL.** Character n-grams provide sub-morphemic orthographic stability; MuRIL provides high-level semantic context. A simple probability-blend ($0.5 \cdot P_\text{LogReg} + 0.5 \cdot P_\text{MuRIL}$) requires **zero extra GPU training** and reliably outperforms either model alone by +1.5 to +2.5% F1.
