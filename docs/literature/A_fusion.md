# Theme A: Dynamic, quality-aware and uncertainty-aware multimodal fusion, and modality imbalance

ARS `lit-review`, Phase 2: annotated bibliography and source verification.
Project: *Uncertainty-aware occlusion-robust RGB-skeleton action recognition on NTU RGB+D 120* (frozen R3D-18 + CTR-GCN; Milestone 4 = M4).
Prepared: 2026-11-08. Scope: Phase 2 only. Synthesis and drafting are left to downstream stages.

---

## 0. Headline positioning facts (for the reviewer response)

1. **TCP / MMDynamics / PDF already use M4 "Target A".** TCP (Corbière et al., 2019) regresses p(y_true) with MSE from a **frozen** classifier's features. MMDynamics (Han et al., 2022) uses a predicted TCP per modality to scale features before concatenation (feature-level fusion). PDF (Cao et al., 2024) trains an MLP to regress p_true with MSE and uses it as a logit-level fusion weight. A Linear-GELU-Dropout-Linear-Sigmoid reliability head regressing p(y_true) per branch is therefore the TCP/PDF recipe applied to RGB+skeleton. It is not a new estimator.
2. **M4 "Target B" exp(-CE) is the same quantity as Target A** when the loss is plain single-label cross-entropy, because exp(-CE) = exp(log p_y) = p_y. The two targets differ only if CE includes label smoothing, a temperature, or a different (e.g. clip- vs segment-level) aggregation. This is an inference from the definitions, not a literature claim. Check what the branches were trained with before presenting them as two targets.
3. **QMF's theory gives M4's AUROC analysis a purpose.** QMF (Zhang et al., 2023, Thm. 1-2) shows dynamic late fusion provably beats static fusion when per-modality fusion weights are **negatively correlated with that modality's loss**. An AUROC of branch uncertainty against branch errors is a direct empirical test of that condition on NTU-120 corruptions. QMF uses an energy score with a ranking regulariser, which is the logit-level analogue of M4's entropy weighting.
4. **The train-vs-val choice for fitting the head has precedent on both sides.** TCP's authors trained ConfidNet on the training set and reported that a 10 % hold-out validation set *lowered* AUPR-Error (few errors to learn from). Gradient-Blending estimates its modality weights from held-out validation loss. Fitting a combiner on held-out predictions is Wolpert's (1992) stacked generalization. M3's "gate trained on train features saturates onto RGB" is consistent with TCP's stated cause: overfit branches leave few training errors.
5. **Training-time corruption awareness already exists.** UDML (Wei et al., CVPR 2026) supervises a per-modality estimator to predict the *intensity of injected noise*. It reports that entropy/confidence-based estimators are insensitive to slight degradation and still give non-negligible weight to heavily corrupted modalities. This directly bears on M4's "clean-trained vs corruption-aware" axis.
6. **A graded-corruption benchmark design is precedented, but not for RGB+skeleton.** READ (Yang et al., ICLR 2024) builds Kinetics50-C / VGGSound-C with 15 video + 6 audio corruptions × 5 severities, but for audio-visual test-time adaptation.
7. **Gap flag: no verified peer-reviewed work does uncertainty/reliability-weighted RGB+skeleton fusion on NTU under graded synthetic corruptions with calibration (ECE/NLL/Brier) or selective-prediction (AURC) evaluation.** Nearest items:
   - AMFI-Net (JVCIR 2025): RGB+skeleton on NTU-60, learned confidence weights, no corruption tests.
   - MAGNet (Sci. Rep. 2026): RGB+depth+skeleton on NTU-60/120, missing-modality tests only, no uncertainty.
   - RCMCL (arXiv 2025, unreviewed, credibility flags): RGB-D+skeleton+point cloud on NTU-120 with a reliability gate and skeleton-noise/dropout tests.
   - An NHSJS 2026 student-journal report: RGB+skeleton probability/log-probability fusion under synthetic occlusion on UCF-101 (not NTU), essentially M3's baseline set, with uncertainty-aware fusion listed as future work.

   This gap claim is bounded by the searches in Section 1.

---

## 1. Search strategy

**Databases / sources:**
- OpenAlex API (title search plus DOI lookups)
- arXiv API (title search, abstracts, author comments)
- Crossref API (DOI metadata)
- Official proceedings pages: CVF Open Access, PMLR, NeurIPS Proceedings, ICLR virtual site / proceedings, PMC
- Web search (10 queries)
- Semantic Scholar API: attempted, returned HTTP 429 on every call → `[S2-API-UNAVAILABLE]`. Tier 0 was skipped and Tiers 1-2 (DOI resolution + venue page / web check) were used instead.
- OpenReview: blocked by a bot challenge. ICLR items were confirmed via the ICLR site, mlanthology and arXiv comments.

**Keywords:**
- Named seed titles (9 required items).
- "dynamic multimodal fusion", "quality-aware fusion", "uncertainty-aware fusion", "true class probability", "evidential / Dempster-Shafer fusion", "modality imbalance", "gradient modulation", "missing modality transformer", "multi-modal test-time adaptation reliability bias", "calibrating multimodal", "RGB skeleton fusion NTU reliability / confidence / occlusion / corruption", "uncertainty-aware activity recognition".

**Boolean (web):** (uncertainty OR reliability OR confidence OR evidential) AND (fusion) AND (RGB AND (skeleton OR pose)) AND (action recognition) [AND (NTU OR occlusion OR corruption)].

**Date range:** seminal works unrestricted (1992-2022); follow-ups 2023-2026.

**Last searched:** 2026-11-08.

**Inclusion criteria:**
- (a) Proposes or analyses per-sample or per-modality reliability/quality/uncertainty weighting in multimodal classification; or
- (b) Diagnoses or fixes modality imbalance; or
- (c) Is RGB+skeleton/pose fusion with adaptive weights or robustness tests; or
- (d) Is a theoretical basis the project's fusion heads rest on (stacking).
- The source must be peer-reviewed, or an arXiv preprint that is uniquely relevant (flagged).

**Exclusion criteria:**
- Unimodal uncertainty only (handled by another theme).
- Wearable/IMU-only HAR.
- Fusion for computational efficiency only, unless widely cited as "dynamic fusion".
- Metadata that could not be confirmed (→ UNVERIFIED list).

**PRISMA-style flow:**

```
Records identified (total): ~128
|-- Web search (10 queries x ~9 hits): ~88
|-- OpenAlex title queries: ~30 hits
+-- arXiv API: 10 hits
Duplicates removed: ~52
Records screened (title/abstract): ~76
Records excluded at screening: ~40 (off-theme: unimodal skeleton uncertainty, TAL-only evidential, medical segmentation, emotion-only preprints, IMU HAR)
Full-text / metadata assessed: 36
Full-text excluded (with reasons): 8 (see Section 5); 3 more could not be verified (Section 6)
Studies included: 25
```

(Web hit counts are approximate, since result pages vary. OpenAlex/arXiv counts are exact.)

**Coverage distribution advisory:**

```
DISTRIBUTIONAL_SKEW_ADVISORY:
- Dimension: methodological distribution
- Concentration: computational/experimental method papers = 22/25 (88%)
- Advisory: Expected for an ML methods review; the remaining 3 are 1 survey, 1 theory paper, 1 student report.
- Search response: no expansion; the RQ is methodological.
```

**Lineage note (not a skew trigger):** 8/25 sources share the Tianjin University (C. Zhang / Q. Hu) lineage: TMC ×2, MMDynamics, QMF, PDF, CML, READ, the 2026 survey. These are the works the reviewers named, so the concentration is expected. However, this lineage's own survey and benchmark protocol (Gaussian noise on 50 % of modalities) define "low-quality" for most comparisons. Treat that as an intellectual COI (Moderate) when citing their framing.

---

## 2. Verification summary (Source Verification Report)

**Sources reviewed:** 28 (25 included + 3 excluded after full-text check for scope). **Verified:** 25. **Flagged:** 2 (RCMCL, NHSJS). **Rejected:** 0 fabricated; 1 mis-described by a search engine (Dong et al., 2025, see Section 5).

**Status key:** VERIFIED = DOI resolves via Crossref/OpenAlex and title/authors match, or the official proceedings page was fetched. PLAUSIBLE = existence confirmed by arXiv plus an index but no official page fetched.

| # | Short ref | Status / route | Level* | Venue | Method (scope) | Currency | COI | Overall |
|---|---|---|---|---|---|---|---|---|
| 1 | Han 2022 MMDynamics | VERIFIED: CVF page + DOI 10.1109/CVPR52688.2022.02005 | III | pass | pass | pass | warn (lineage) | A |
| 2 | Zhang 2023 QMF | VERIFIED: PMLR v202 page | III | pass | pass | pass | warn (lineage) | A |
| 3 | Han 2021 TMC | VERIFIED: arXiv comment + mlanthology ICLR 2021 + OpenReview PDF listing | III | pass | abstract only | pass | warn | A- |
| 4 | Han 2023 TMC-TPAMI | VERIFIED: Crossref DOI 10.1109/TPAMI.2022.3171983 | III | pass | abstract only | pass | warn | A- |
| 5 | Cao 2024 PDF | VERIFIED: PMLR v235 page | III | pass | pass | pass | warn | A |
| 6 | Corbière 2019 TCP | VERIFIED: NeurIPS proceedings page + arXiv | III | pass | pass | foundational | pass | A |
| 7 | Wang 2020 G-Blend | VERIFIED: CVF + DOI | III | pass | pass | foundational | pass | A |
| 8 | Wu 2022 Greedy | VERIFIED: PMLR v162 page | III | pass | abstract + Alg. 1 | pass | pass | A |
| 9 | Peng 2022 OGM-GE | VERIFIED: CVF + DOI | III | pass | pass | pass | pass | A |
| 10 | Wei 2025 OPM/OGM TPAMI | VERIFIED: Crossref DOI | III | pass | abstract only | pass | pass | A- |
| 11 | Wolpert 1992 | VERIFIED: Crossref DOI | VII (theory) | pass | metadata only | foundational | pass | A- |
| 12 | Ma 2023 CML | VERIFIED: PMLR v202 page | III | pass | pass | pass | warn | A |
| 13 | Yang 2024 READ | VERIFIED: ICLR virtual page + ICLR proceedings PDF | III | pass | pass | pass | warn | A |
| 14 | Wei 2026 UDML | VERIFIED: OpenAlex DOI 10.1109/CVPRF71909.2026.00599 (CVPR 2026, pp. 6242-6252) + arXiv | III | pass | pass | pass | pass | A |
| 15 | Ma 2022 missing-modality Transformers | VERIFIED: CVF page + DOI | III | pass | abstract only | pass | pass | A- |
| 16 | Du 2023 UME/UMT | VERIFIED: PMLR v202 page | III | pass | abstract only | pass | pass | A- |
| 17 | Gao 2024 EAU | VERIFIED: OpenAlex DOI + CVF listing | III | pass | abstract only | pass | pass | A- |
| 18 | Subedar 2019 | VERIFIED: CVF ICCV page + DOI | III | pass | abstract only | foundational | warn (industry lab, Intel) | A- |
| 19 | Xue 2023 DynMM | VERIFIED: OpenAlex DOI (CVPRW 2023) | III | pass | abstract only | pass | pass | B+ |
| 20 | Zhang 2026 survey | VERIFIED: Crossref DOI (Inf. Fusion 135) + arXiv 2404.18947 | V | pass | abstract only | pass | warn (lineage) | A- |
| 21 | Xu 2025 AMFI-Net | VERIFIED: OpenAlex DOI + Essex repository full text | III | pass | sections | pass | pass | B |
| 22 | Dong 2026 MAGNet | VERIFIED: OpenAlex DOI + PMC full text | III | pass (Sci. Rep.) | warn (internal inconsistencies) | pass | pass | B- |
| 23 | Takami 2026 | VERIFIED: PMC page, DOI 10.3390/s26082454 | III | pass (MDPI Sensors, indexed) | pass | pass | pass | B |
| 24 | Akgul 2025 RCMCL | PLAUSIBLE: arXiv 2511.04351v3 exists; no venue | VI | **warn** (preprint, eess.SP) | warn | pass | n/a | **C, flagged** |
| 25 | Guo 2026 NHSJS | Page exists (nhsjs.com); no DOI | VI | **warn** (high-school journal; peer review of this item unconfirmed) | warn | pass | n/a | **C, flagged** |

*Evidence levels are mapped from the 7-level hierarchy onto ML research: III = controlled benchmark comparison; V = survey; VI = single small or unreviewed study; VII = theory/expert framing.

**Flagged sources:**
- **RCMCL (Akgul et al., arXiv 2511.04351).**
  - Issue: not peer-reviewed. Six authors at six institutions in five countries, with generic names and no group lineage found. Figure text partly garbled in the PDF.
  - Severity: High for citation as evidence.
  - Recommendation: cite only as "a preprint claims…", never as an established result.
- **Guo (2026, NHSJS).**
  - Issue: high-school journal, single author, one UCF-101 split, one occlusion seed, no DOI.
  - Severity: Medium.
  - Recommendation: cite as a non-archival precedent for the baseline design (it overlaps M3 closely), not as evidence.
- **MAGNet (Dong et al., 2026).** The WebFetch reading reported internal inconsistencies: Table 3 mentions diffusion imputation while the method describes a VQ-VAE; class count for UTD-MHAD differs between sections. Recommendation: include with caveat.
- **Predatory-journal alerts:** none. MDPI *Sensors* and *Scientific Reports* are indexed; no predatory flags apply.

---

## 3. Annotated bibliography (N = 25)

Each entry gives:
- **Read scope**
- **WHY** (problem)
- **HOW** (reliability quantity, supervision, level, per-sample/per-time)
- **WHAT** (results, datasets, corruptions)
- **Relation to M4**
- **Method weaknesses**, following the bibliography-agent rules. Provenance tags: [AA] = author-acknowledged, [RI] = reader-inferred.

### 3.1 Learned-confidence and dynamic late fusion (the papers the panel named)

**1. Han, Z., Yang, F., Huang, J., Zhang, C., & Yao, J. (2022).** Multimodal dynamics: Dynamical fusion for trustworthy multimodal classification. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 20675-20685). IEEE. https://doi.org/10.1109/CVPR52688.2022.02005 (CVF open access: pp. 20707-20717)
- **Read scope:** abstract; Sec. 3.1-3.3 (Eqs. 1-7); Sec. 4 datasets and compared methods (CVF PDF).
- **WHY:** Existing multimodal classifiers exploit complementarity but do not assess, per sample, how informative each feature and each modality is. This matters for trustworthy (medical) decisions.
- **HOW:**
  - Two levels of weighting:
    - *Feature level:* a sparse gate w^m = σ(E^m(x^m)) with an ℓ1 penalty.
    - *Modality level:* a per-modality confidence network regresses **TCP^m = p^m(y_true | x^m)** with MSE (L_conf = Σ_m (TCP-hat^m − TCP^m)² + L_cls).
  - The fused representation concatenates each modality's features **multiplied by its predicted TCP**, then a classifier is trained with CE.
  - Weighting is per-sample, per-modality. There is no temporal dimension. Everything is trained jointly end-to-end with the encoders.
- **WHAT:** Four multi-omics medical datasets (BRCA, LGG, ROSMAP, KIPAN; TCGA). Beats TMC, GMU, MOGONET etc. (e.g. BRCA ACC 87.7 %). **No synthetic-corruption benchmark in the CVPR paper** (checked: Sec. 4 experiments and tables).
- **Relation to M4:** MMDynamics' modality-level target = TCP = **M4 Target A**. Its fusion (TCP-scaled features → concat → classifier) is the **feature-level uncertainty-guided fusion** M4 plans. Differences:
  - M4 branches are frozen video/skeleton networks rather than jointly trained MLP encoders.
  - M4 adds 8 temporal segments with per-segment reliability, which MMDynamics lacks.
  - M4 evaluates under graded corruptions plus calibration/AURC, which MMDynamics does not.
- **Method weaknesses** (empirical; read: Secs. 3-4):
  - Checked, found: datasets, baselines, ablation. Checked, none found: corruption/robustness tests, calibration metrics. Not checked: statistical tests, seeds.
  - TCP regression is trained on the same training data as the classifiers. When the per-modality classifiers fit the training set, the TCP targets cluster near 1 and the confidence net gets few low-TCP examples. This distorts weighting at test time when errors are common. [RI]
  - Small datasets (351-875 samples). Gains of ~1 point may be within seed variance. [RI]

**2. Zhang, Q., Wu, H., Zhang, C., Hu, Q., Fu, H., Zhou, J. T., & Peng, X. (2023).** Provable dynamic fusion for low-quality multimodal data. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 41753-41769). PMLR. https://proceedings.mlr.press/v202/zhang23ar.html (arXiv: 2306.02050)
- **Read scope:** abstract; Sec. 3 (Thm. 1 generalization bound); Sec. 4 (energy-score uncertainty, ranking regulariser Eqs. 13-18); Sec. 5 setup and Table 1; App. C.2.
- **WHY:** Dynamic fusion is widely used but lacks theory. When is dynamic weighting provably better than static weighting?
- **HOW:**
  - **Theory:** for decision-level fusion f = Σ w^m f^m, the generalization bound contains a term Cov(w^m, ℓ^m). Dynamic beats static when the weights are **negatively correlated with unimodal loss**.
  - **Method (QMF):** the weight comes from an **energy score** of each unimodal classifier's logits, −T·log Σ_k exp(f_k/T), i.e. a logit-derived uncertainty.
  - Supervision: no ground-truth target. Instead a **sample-ranking regulariser** L_reg enforces that samples with higher historical loss get lower weight.
  - Logit (decision) level, per-sample, no temporal component. Trained jointly with unimodal CE losses.
- **WHAT:**
  - Datasets: NYU Depth V2 and SUN RGB-D (RGB+depth); UPMC Food-101 and MVSA (image+text).
  - Corruptions: Gaussian noise (image) and blank noise (text) on **50 % of modalities** at several σ; salt-and-pepper in the appendix.
  - Metrics: average and worst-case accuracy over 10 seeds. QMF is usually in the top 3.
  - The paper also reports that accuracy tracks uncertainty-estimation quality (Pearson correlation, Fig. 3).
- **Relation to M4:**
  - M4's entropy-weighted logit fusion is QMF's design with entropy in place of energy and without L_reg.
  - M4's **AUROC of uncertainty vs branch errors** is the empirical check of QMF's Cov(w, ℓ) < 0 condition. Framing it that way turns an overlap into a test of QMF's theory on a new modality pair.
  - QMF does not use temporal segments, a learned regression head, or corruption-aware training.
- **Method weaknesses** (empirical; read: Secs. 3-5, App. C):
  - Checked, found: seeds (10), worst-case accuracy, ablation of L_reg. Checked, none found: calibration metrics (ECE/NLL). Not checked: compute cost.
  - Noise is applied to 50 % of modalities with i.i.d. Gaussian pixels. This distorts conclusions when real degradations are structured (occlusion, missing joints), which is M3's setting. [RI]

**3. Han, Z., Zhang, C., Fu, H., & Zhou, J. T. (2021).** Trusted multi-view classification. In *International Conference on Learning Representations*. https://openreview.net/forum?id=OOsR8BzCnl5 (arXiv: 2102.02051)
- **Read scope:** abstract only.
- **WHY:** Multi-view classifiers should assess view quality per sample and give trustworthy uncertainty.
- **HOW:**
  - Each view outputs non-negative **evidence**, which parameterises a Dirichlet over class probabilities (subjective logic: belief masses + uncertainty mass u = K/S).
  - Views are combined with **Dempster-Shafer's rule**, so a view with high uncertainty mass contributes less.
  - Evidence level, per-sample. Supervised only through the classification (evidential) loss; there is no explicit reliability target.
- **WHAT:** Abstract reports gains in accuracy, reliability and OOD robustness. Datasets are not stated in the abstract.
- **Relation to M4:** An evidential alternative to M4's entropy and p(y_true) reliabilities. It would require retraining branch heads to output evidence, which conflicts with the frozen-branch design. It is useful as a named baseline family ("evidence-level fusion").
- **Method weaknesses:** not assessed (read scope: abstract_only).

**4. Han, Z., Zhang, C., Fu, H., & Zhou, J. T. (2023).** Trusted multi-view classification with dynamic evidential fusion. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*(2), 2551-2566. https://doi.org/10.1109/TPAMI.2022.3171983 (arXiv: 2204.11423; online 2022)
- **Read scope:** abstract only.
- **WHY:** Extends TMC to noisy, corrupted and OOD data.
- **HOW:** Variational Dirichlet per view + Dempster-Shafer combination, per sample, at evidence level. The abstract explicitly targets robustness to "noise or corruption".
- **WHAT:** Theoretical and experimental validation of accuracy, robustness and trustworthiness (details not read).
- **Relation to M4:** Same as #3. This is the TPAMI extension the brief asked about, and it exists.
- **Method weaknesses:** not assessed (read scope: abstract_only).

**5. Cao, B., Xia, Y., Ding, Y., Zhang, C., & Hu, Q. (2024).** Predictive dynamic fusion. In *Proceedings of the 41st International Conference on Machine Learning* (Vol. 235, pp. 5608-5628). PMLR. https://proceedings.mlr.press/v235/cao24c.html (arXiv: 2406.04802)
- **Read scope:** abstract; Sec. 1; Sec. 4 (Eqs. 7-11); App. C.4 and C.6 (predictor and loss).
- **WHY:** Dynamic fusion methods lack guarantees and are unstable. Build weights that provably reduce the generalization bound.
- **HOW:**
  - A **confidence predictor (stacked linear layers) regresses p_true^m from modality features with MSE** (App. C.6, Eq. 30).
  - *Co-Belief* = **Mono-Confidence** (own predicted p_true, which should covary negatively with own loss) + **Holo-Confidence** (built from the other modality's predicted p_true, which should covary positively with the other modality's loss).
  - **Relative Calibration:** distribution uniformity DU^m = mean_i |softmax_i − 1/C|. The ratio RC^m = DU^m/DU^n down-weights the less peaked modality (asymmetric, only when RC < 1).
  - Final weights = softmax over calibrated Co-Beliefs, applied to unimodal outputs. Logit/decision level, per-sample, no temporal component.
- **WHAT:** MVSA, Food-101, NYU Depth V2, CREMA-D (+ PIE), with Gaussian noise on 50 % of modalities (image/audio) and blank noise (text), 5 seeds. Outperforms QMF and late fusion. Adds covariance-based metrics.
- **Relation to M4:**
  - PDF's predictor target = **p(y_true) = M4 Target A**. Its head (MLP on branch features, MSE) is structurally the M4 reliability head.
  - M4 does **not** yet have PDF's cross-modal "Holo" term or relative calibration. M4 differs by per-segment temporal reliability, temperature scaling before weighting, and calibration/AURC evaluation.
  - This is the closest single precedent for M4's logit-level design. It should be a named baseline or at least cited as the origin of the regression-to-p_true fusion weight.
- **Method weaknesses** (empirical; read: Sec. 4, App. C):
  - Checked, found: seeds (5), multiple datasets. Checked, none found: ECE/NLL of the fused model in the sections read. Not checked: main-paper ablations.
  - p_true is regressed on training data from jointly trained encoders. The same near-1 target collapse as in #1 applies when unimodal heads overfit. [RI]

**6. Corbière, C., Thome, N., Bar-Hen, A., Cord, M., & Pérez, P. (2019).** Addressing failure prediction by learning model confidence. In *Advances in Neural Information Processing Systems* (Vol. 32). Curran Associates. https://papers.nips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html (arXiv: 1910.04851)
- **Read scope:** abstract; Sec. 2 (TCP criterion, ConfidNet, learning scheme); Sec. 3.1 metrics; Sec. 3.3 (Tables 3-4).
- **WHY:** Maximum class probability (MCP) is a poor failure detector because it is overconfident on errors.
- **HOW:**
  - Target = **True Class Probability**, the softmax value at the ground-truth class (also a normalised TCP_r).
  - **ConfidNet** is a small dense head on the **frozen** classifier's features, trained with **MSE** to TCP. The encoder is then optionally fine-tuned as a decoupled copy, with dropout off.
  - Per-sample, unimodal.
- **WHAT:**
  - Datasets: MNIST, SVHN, CIFAR-10/100, CamVid (segmentation). Metrics: AUPR-Error (main), AUPR-Success, FPR@95TPR, AUROC. Beats MCP, Trust Score and MC-dropout.
  - **Table 3:** training ConfidNet on a 10 % **held-out validation set lowers performance**, because high-accuracy models leave few validation errors.
  - **Table 4:** MSE beats BCE.
- **Relation to M4:**
  - TCP target = **identical to M4 Target A**. ConfidNet-on-frozen-classifier = **identical setting to M4's reliability head on frozen R3D-18/CTR-GCN**. M4's AUROC of uncertainty vs errors corresponds to TCP's failure-prediction metrics, and AURC extends them.
  - M4's decision to fit on validation (because train-fitted heads saturate) is the variant TCP found weaker on small data. NTU-120's validation split and corruption-augmented errors may reverse that finding, and reporting both would be informative.
  - M4's new elements relative to TCP: two modalities, per-segment reliability, and using the predicted TCP as a fusion weight rather than only a failure score.
- **Method weaknesses** (empirical; read: Secs. 2-3):
  - Checked, found: multiple datasets and architectures, loss ablation, validation-set ablation. Checked, none found: distribution-shift/corruption evaluation. Not checked: supplementary material.
  - [AA, section 3.3] The small number of training errors under overfitting limits what ConfidNet can learn ("Given the small number of errors available due to deep neural network over-fitting…").

**7. Wang, W., Tran, D., & Feiszli, M. (2020).** What makes training multi-modal classification networks hard? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 12692-12702). IEEE. https://doi.org/10.1109/CVPR42600.2020.01271 (arXiv: 1905.12681)
- **Read scope:** abstract; Sec. 1; Sec. 2 (OGR definition, Algorithm); Sec. 3.1; Table 1.
- **WHY:** Late-fusion multimodal video networks underperform the best unimodal network (e.g. Kinetics RGB 72.6 vs A+RGB 71.4).
- **HOW:**
  - Modalities overfit and generalise at different rates.
  - **Gradient-Blending** estimates per-modality loss weights from the **overfitting-to-generalization ratio**, using a **held-out validation loss** as a proxy for the true loss. These weights blend unimodal and multimodal losses during training.
  - Global per-modality weights (not per-sample), loss level, offline or online.
- **WHAT:** Kinetics, mini-Sports, mini-AudioSet ablations. State of the art on Kinetics, EPIC-Kitchens and AudioSet with RGB, optical flow and audio.
- **Relation to M4:** Precedent for **using a held-out split to set fusion weights**, which is what M3 did. It also shows imbalance in a video action-recognition setting. Its weights are global and set at training time, unlike M4's per-sample, per-segment reliability. It is not uncertainty-based.
- **Method weaknesses** (empirical; read: Secs. 1-3):
  - Checked, found: ablations, multiple datasets. Not checked: seeds/variance.
  - OGR is estimated from validation loss. Weights become noisy when the validation split is small or distribution-shifted relative to test (e.g. corrupted test, clean validation). [RI]

**8. Wu, N., Jastrzębski, S., Cho, K., & Geras, K. J. (2022).** Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In *Proceedings of the 39th International Conference on Machine Learning* (Vol. 162, pp. 24043-24055). PMLR. https://proceedings.mlr.press/v162/wu22d.html (arXiv: 2202.05306)
- **Read scope:** abstract; Algorithm 1.
- **WHY:** Multimodal networks greedily rely on one modality and under-fit the others, which hurts generalization.
- **HOW:**
  - Defines the **conditional utilization rate**: the accuracy gain from adding a modality.
  - Proxies it during training with the **conditional learning speed** and re-balances with alternating "re-balancing steps".
  - Training-time intervention, not a per-sample weight.
- **WHAT:** Colored MNIST, ModelNet40, NVIDIA Dynamic Hand Gesture (RGB + depth). Improved generalization.
- **Relation to M4:** Explains M3's diagnostic finding: a gate trained on training features **saturated onto RGB**, a fusion-head form of greedy reliance on the modality that fits the training set fastest [RI]. M4's frozen branches avoid encoder-level greed. A learned gate or reliability head can still be greedy, so reporting the conditional utilization rate per corruption would connect to this literature.
- **Method weaknesses:** abstract and algorithm only, so not assessed (read scope: abstract_only + algorithm).

**9. Peng, X., Wei, Y., Deng, A., Wang, D., & Hu, D. (2022).** Balanced multimodal learning via on-the-fly gradient modulation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 8228-8237). IEEE. https://doi.org/10.1109/CVPR52688.2022.00806 (arXiv: 2203.15332)
- **Read scope:** abstract; Sec. 3.2 (discrepancy ratio); Sec. 4.1 datasets.
- **WHY:** A dominant modality (e.g. audio) leaves the other under-optimised even when the fused model beats unimodal ones.
- **HOW:**
  - Discrepancy ratio = ratio of summed true-class softmax scores of the two unimodal heads over a mini-batch.
  - Gradients of the dominant modality are scaled down, and Gaussian noise is added for generalization (GE).
  - Batch-level, training-time.
- **WHAT:** CREMA-D, Kinetics-Sounds, VGGSound, AVE (audio-visual). Consistent gains over concat/sum/FiLM fusion.
- **Relation to M4:** OGM-GE's contribution signal is a **batch-summed true-class probability**, the batch-level relative of TCP/Target A. With frozen branches, OGM-GE itself does not apply. It is cited to separate *imbalance during joint training* (not M4's problem) from *per-sample reliability at inference* (M4's problem).
- **Method weaknesses** (empirical; read: Secs. 3.2, 4.1): Checked, found: 4 datasets. Not checked: seeds. Not otherwise assessed.

**10. Wei, Y., Hu, D., Du, H., & Wen, J.-R. (2025).** On-the-fly modulation for balanced multimodal learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 47*(1), 469-485. https://doi.org/10.1109/TPAMI.2024.3468315
- **Read scope:** abstract only.
- **WHY / HOW:** Journal extension of #9. Adds **On-the-fly Prediction Modulation (OPM)**, which drops the dominant modality's feature with a dynamic probability in the forward pass, alongside OGM in the backward pass.
- **WHAT:** Gains across vanilla, task-oriented and complex multimodal tasks (details not read).
- **Relation to M4:** OPM's dynamic modality dropout during training is related to M4's corruption-aware training (skeleton body-part/joint dropout and temporal masking), but OPM is driven by imbalance rather than reliability.
- **Method weaknesses:** not assessed (read scope: abstract_only).

**11. Wolpert, D. H. (1992).** Stacked generalization. *Neural Networks, 5*(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1
- **Read scope:** bibliographic metadata only (Crossref). Content is described from the standard account of stacking.
- **WHY / HOW:** A level-1 combiner is trained on level-0 models' predictions for data **not used to fit those level-0 models** (held-out / cross-validated predictions), so the combiner learns how to correct the base learners' biases.
- **WHAT:** Theoretical/methodological paper.
- **Relation to M4:**
  - M3's fusion heads (concat, global weight, attention gate) fit on the validation split's frozen-branch log-probabilities are stacked generalization.
  - The same logic covers M4's temperature scaling and reliability head. Stacking theory also implies that fitting several stages on one held-out split (temperature, then reliability head, then gate) reuses it, so a cross-fitted or nested split protects test-set claims [RI].
- **Method weaknesses:** not assessed (read scope: metadata_only).

### 3.2 Calibration, test-time reliability and corruption-aware weighting (2023-2026 follow-ups)

**12. Ma, H., Zhang, Q., Zhang, C., Wu, B., Fu, H., Zhou, J. T., & Hu, Q. (2023).** Calibrating multimodal learning. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 23429-23450). PMLR. https://proceedings.mlr.press/v202/ma23i.html (arXiv: 2306.01265)
- **Read scope:** abstract; Sec. 3 (principle, Eq. 6); Sec. 4.1 setup.
- **WHY:** Multimodal classifiers' confidence can *increase* when a modality is corrupted. The models rely on partial modalities.
- **HOW:** Principle: **confidence should not increase when a modality is removed**. The **CML regulariser** penalises violations sample-wise and is added to CPM-Nets, MIWAE, MMTM, etc.
- **WHAT:** Improved calibration, accuracy and robustness on two-modality datasets, including incomplete-modality settings.
- **Relation to M4:** Supplies a **calibration diagnostic M4 can report at no cost**: the fraction of NTU test samples whose fused confidence exceeds a single-branch confidence when the other branch is corrupted. It complements ECE/NLL/Brier. M4's temperature scaling is per-branch and post hoc, while CML is a training-time regulariser.
- **Method weaknesses** (empirical; read: Secs. 3-4.1): Not checked: results tables. Partially assessed only.

**13. Yang, M., Li, Y., Zhang, C., Hu, P., & Peng, X. (2024).** Test-time adaptation against multi-modal reliability bias. In *The Twelfth International Conference on Learning Representations*. https://iclr.cc/virtual/2024/poster/18574 (PDF: https://proceedings.iclr.cc/paper_files/paper/2024/file/f25d75fc760aec0a6174f9f5d9da59b8-Paper-Conference.pdf)
- **Read scope:** abstract; Sec. 2; Fig. 2; Sec. 4 benchmark construction; Table 1 caption.
- **WHY:** When one modality shifts at test time (corruption), it loses task information and biases fusion ("reliability bias"). Existing TTA methods are unimodal and normalisation-based.
- **HOW:**
  - **READ** adapts the attention-fusion layers of a frozen-encoder CAV-MAE at test time (self-adaptive cross-modal attention).
  - A confidence-aware objective amplifies confident predictions and suppresses noisy ones.
  - Reliability is implicit (attention), per-sample and token-level, and unsupervised at test time.
- **WHAT:** **Kinetics50-C and VGGSound-C**: 15 video + 6 audio corruptions (Hendrycks-Dietterich style) × **5 severities**, corrupting one modality at a time. READ beats TTA baselines (Tent, EATA, SAR, MM-TTA) at severity 5.
- **Relation to M4:** The closest **benchmark-design** precedent for M3 (graded corruptions on one modality of an action-recognition pair, severity 1-5), but audio-visual and test-time adaptive. M4 instead uses a supervised reliability head and post-hoc calibration, with no test-time parameter updates. Cite it when positioning M3's corruption protocol.
- **Method weaknesses** (empirical; read: Secs. 2, 4):
  - Checked, found: 5 seeds, multiple corruption types and severities. Checked, none found: calibration metrics in the sections read.
  - Corruptions are synthetic and imported from image-domain types. This distorts relevance when real failures are structured (person-box occlusion, skeleton tracking loss), which is the gap M3 targets. [RI]

**14. Wei, S., Zhang, K., Chen, L., He, T., & Duan, G. (2026).** Unbiased dynamic multimodal fusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 6242-6252). https://doi.org/10.1109/CVPRF71909.2026.00599 (arXiv: 2603.19681)
- **Read scope:** abstract; Sec. 1 contributions; Sec. 3 noise-aware estimator (Eqs. 9-10); Sec. 4.1 datasets.
- **WHY:**
  - Dynamic fusion relies on empirical uncertainty metrics that fail at very low or very high noise.
  - Equal initial modality contributions ignore inherent modality-dependency bias, so a hard-to-learn modality is "doubly penalised" and dynamic fusion can underperform static fusion.
- **HOW:**
  - **Noise-aware uncertainty estimator:** each modality is encoded as a Gaussian (μ, σ). An estimator is **supervised to regress the intensity of Gaussian noise injected into the input**, from the variance term, using MSE over a discrete set of noise levels.
  - **Modality-dependency calculator:** uses modality dropout to measure inherent reliance and re-calibrates the weights.
  - Feature level, per-sample. Trained progressively with the task.
- **WHAT:** CMU-MOSI, CMU-MOSEI, MVSA-Single, CREMA-D and one more benchmark. The authors report that existing uncertainty estimators are insensitive to slight degradation and assign non-negligible weight to heavily corrupted modalities.
- **Relation to M4:**
  - The strongest 2026 precedent for **corruption-aware reliability training**: the target is *corruption severity*, whereas M4 targets *correctness*.
  - Its "double penalty" argument is a concrete risk for skeleton vs RGB on NTU: CTR-GCN is the stronger branch on clean data, while RGB may be the "hard" modality.
  - Its finding about estimator insensitivity predicts what M4's entropy-AUROC analysis may show at low severities.
- **Method weaknesses** (empirical; read: Secs. 1, 3, 4.1):
  - Checked, found: 5 datasets. Not checked: results tables, seeds.
  - The estimator is trained on additive Gaussian noise only. It may not transfer to occlusion or dropout-type corruptions unless trained on them. [RI]

**15. Ma, M., Ren, J., Zhao, L., Testuggine, D., & Peng, X. (2022).** Are multimodal transformers robust to missing modality? In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 18177-18186). https://doi.org/10.1109/CVPR52688.2022.01764 (arXiv: 2204.05454)
- **Read scope:** abstract only (CVF page).
- **WHY:** Are multimodal Transformers robust when a modality is missing at test time?
- **HOW:** Empirical study of fusion strategies, plus an automatic search for the best fusion strategy per dataset.
- **WHAT:** Transformers are sensitive to missing modalities, and **the optimal fusion strategy is dataset-dependent**. Evaluated on three benchmarks.
- **Relation to M4:** Missing-modality is the limit case of M3's severity-5 corruptions (full RGB occlusion / full skeleton dropout). Supports reporting M4's fusion heads per corruption family rather than one aggregate.
- **Method weaknesses:** not assessed (read scope: abstract_only).

**16. Du, C., Teng, J., Li, T., Liu, Y., Yuan, T., Wang, Y., Yuan, Y., & Zhao, H. (2023).** On uni-modal feature learning in supervised multi-modal learning. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 8632-8656). PMLR. https://proceedings.mlr.press/v202/du23e.html
- **Read scope:** abstract only (PMLR page).
- **WHY:** Joint late-fusion training under-learns uni-modal features.
- **HOW:** Chooses between a **Uni-Modal Ensemble (UME)**, i.e. independently trained unimodal models combined at the output, and a Uni-Modal Teacher (UMT), based on the balance of uni-modal vs paired features.
- **WHAT:** UME/UMT are comparable to complex late/intermediate fusion on VGG-Sound, **Kinetics-400, UCF101**, ModelNet40.
- **Relation to M4:** Directly justifies the **frozen independently trained branches + output-level fusion** design. M4 is a UME with reliability-weighted rather than uniform combination.
- **Method weaknesses:** not assessed (read scope: abstract_only).

**17. Gao, Z., Jiang, X., Xu, X., Shen, F., Li, Y., & Shen, H. T. (2024).** Embracing unimodal aleatoric uncertainty for robust multimodal fusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 26866-26875). https://doi.org/10.1109/CVPR52733.2024.02538
- **Read scope:** abstract only.
- **WHY:** Unimodal embeddings contain noise that corrupts cross-modal interaction.
- **HOW:** Quantifies **aleatoric uncertainty** per unimodal embedding and uses it in contrastive learning (SUFA) to stabilise features. An information-theoretic integration (RMFI) produces a compact joint representation. Feature level.
- **WHAT:** Five datasets (video, RGB image, text, audio, depth). More noise-resistant than prior fusion. New state of the art on several benchmarks.
- **Relation to M4:** Feature-level, aleatoric alternative to M4's feature-level uncertainty-guided fusion. It requires encoder training, so it is not directly compatible with frozen branches.
- **Method weaknesses:** not assessed (read scope: abstract_only).

**18. Subedar, M., Krishnan, R., Lopez Meyer, P., Tickoo, O., & Huang, J. (2019).** Uncertainty-aware audiovisual activity recognition using deep Bayesian variational inference. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 6301-6310). https://doi.org/10.1109/ICCV.2019.00640 (arXiv: 1811.10811)
- **Read scope:** abstract only (CVF page).
- **WHY:** Multimodal activity recognition ignores per-modality predictive uncertainty.
- **HOW:** Bayesian DNNs (mixed deterministic and variational layers). **Uncertainty-aware multimodal Bayesian fusion** of audio and visual streams. Logit/predictive level.
- **WHAT:** Moments-in-Time subset: PR-AUC +10.2 % over a non-Bayesian baseline. Better confidence than MC dropout. OOD detection on UCF101/MiT. Evaluates OOD, not corruptions.
- **Relation to M4:** The earliest verified **uncertainty-aware fusion for action recognition**, but audio-visual and Bayesian. M4 differs by a cheap entropy/temperature/regression-head approach on frozen branches, and by graded corruptions instead of OOD.
- **Method weaknesses:** not assessed (read scope: abstract_only).

**19. Xue, Z., & Marculescu, R. (2023).** Dynamic multimodal fusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops* (pp. 2575-2584). https://doi.org/10.1109/CVPRW59228.2023.00256
- **Read scope:** abstract only.
- **WHY / HOW:** A gating function makes modality-level or fusion-level routing decisions per input, with a resource-aware loss. The goal is data-dependent forward paths for **efficiency**.
- **WHAT:** −46.5 % compute on CMU-MOSEI with negligible accuracy loss. Better segmentation with 21 % savings on NYU Depth V2.
- **Relation to M4:** Often cited as "DynMM" in dynamic-fusion comparisons (PDF compares against it). Its gate is **not supervised by reliability**. That makes it a useful contrast showing that M3's attention gate, also unsupervised, behaves like DynMM-style gating rather than reliability-aware fusion.
- **Method weaknesses:** not assessed (read scope: abstract_only).

**20. Zhang, Q., Wei, Y., Han, Z., Fu, H., Peng, X., Deng, C., Hu, Q., Xu, C., Wen, J., Hu, D., & Zhang, C. (2026).** Multimodal fusion on low-quality data: A comprehensive survey. *Information Fusion, 135*, 104437. https://doi.org/10.1016/j.inffus.2026.104437 (arXiv: 2404.18947)
- **Read scope:** abstract only (arXiv). Journal metadata via Crossref. The author order differs slightly between arXiv and journal; the journal order from OpenAlex is used here.
- **WHY / HOW:** Data-centric taxonomy of four challenges: (1) noisy, (2) incomplete, (3) imbalanced, (4) **quality-varying** multimodal data.
- **WHAT:** Survey and open problems.
- **Relation to M4:** Gives the reviewer-facing vocabulary. M3/M4 sit in "noisy" (blur, Gaussian joint noise), "incomplete" (occlusion, dropout, temporal mask) and "quality-varying" (per-sample, per-segment reliability). Cite it as the umbrella reference in positioning.
- **Method weaknesses:** not assessed (read scope: abstract_only).

### 3.3 RGB+skeleton/pose fusion with adaptive weights (closest domain matches)

**21. Xu, Z., Tang, C., Hu, H., Wang, W., Qiao, S., & Tong, A. (2025).** Attention mechanism based multimodal feature fusion network for human action recognition. *Journal of Visual Communication and Image Representation, 110*, 104459. https://doi.org/10.1016/j.jvcir.2025.104459 (accepted manuscript: https://repository.essex.ac.uk/40769/)
- **Read scope:** abstract; contributions; Sec. 3.3.3 (feature confidence); Sec. 4.1 datasets.
- **WHY:** RGB+skeleton HAR under-exploits skeleton motion correlation and over-relies on RGB spatial and background content.
- **HOW:**
  - AMFI-Net preprocesses into differential motion representations, uses heterogeneous RGB/skeleton networks and attention fusion.
  - A **feature-confidence module dynamically weights the RGB and skeleton features** ("according to the reliability of the modality").
  - The weights are learned end-to-end from the classification loss only. There is **no correctness-supervised target**.
  - Feature level, per-sample, no temporal reliability.
- **WHAT:** NTU RGB+D 60 and UTD-MHAD. Accuracy gains. **No occlusion/corruption experiments** (searched full text for occlusion/corruption/dropout/missing: none in experiments).
- **Relation to M4:** The verified nearest RGB+skeleton "reliability-weighted" fusion on NTU. Its reliability is an unsupervised attention weight, M3's attention-gate baseline in effect. M4 adds a supervised reliability target, calibration and corruption evaluation.
- **Method weaknesses** (empirical; read: Secs. 3.3.3, 4.1):
  - Checked, none found: robustness tests, calibration. Not checked: seeds.
  - Only NTU-60 (no NTU-120), one protocol reported in the sections read. [RI]

**22. Dong, H., Zhang, F., Liu, C., & Mu, L. (2026).** MAGNet: Enhancing action recognition with multimodal fusion and adaptive graph convolution. *Scientific Reports, 16*, 15100. https://doi.org/10.1038/s41598-026-45601-2
- **Read scope:** full text via PMC (automated reading of ~100k of 136k characters).
- **WHY:** RGB fails under low light and occlusion. Combine RGB, depth and 3D pose robustly.
- **HOW:**
  - A per-modality "quality score" with softmax gives adaptive weights in a multimodal adaptive GCN, plus cross-modal attention and VQ-VAE imputation of missing modalities.
  - A variance regulariser prevents one modality dominating.
  - The paper does **not** specify how the quality score is computed or supervised. Feature level. No uncertainty estimation.
- **WHAT:** NTU RGB+D 60/120, UTD-MHAD. On NTU: full 95.2 % acc, no-RGB 91.5, no-skeleton 89.6, random 30 % modality loss 88.9. A ~4.2 % drop on NTU-120 X-Set with >70 % joints lost. No ECE or uncertainty metrics.
- **Relation to M4:** The only verified peer-reviewed NTU-120 RGB(+depth)+skeleton paper with **missing-modality/joint-loss robustness tables**. It is not uncertainty-aware and not calibrated, and its corruptions are not graded over 5 severities.
- **Method weaknesses** (empirical; read: full text, partial):
  - Checked, found: 5 seeds, effect sizes. Checked, found: inconsistencies (Table 3 mentions diffusion imputation versus the VQ-VAE described in the method; UTD-MHAD class count 27 vs 20).
  - The quality score is unspecified, so the reliability mechanism cannot be reproduced. [RI]

**23. Takami, M., & Fukuda, T. (2026).** Confidence-aware gated multimodal fusion for robust temporal action localization in occluded environments. *Sensors, 26*, 2454. https://doi.org/10.3390/s26082454
- **Read scope:** full text via PMC (automated reading of ~100k of 136k characters).
- **WHY:** Occlusion-corrupted skeletons harm RGB features when fused indiscriminately.
- **HOW:**
  - The **OpenPose per-frame mean joint confidence** (not learned) is added as a log-bias to attention logits in a Skeleton Refinement Module (**per-frame, i.e. temporal reliability**).
  - A learned sigmoid gate (1D conv over concatenated features, initialised mostly closed) does the fusion.
  - Feature level.
- **WHAT:** THUMOS14 and IKEA ASM (natural occlusion). Gated SRM ≈ RGB-only (+0.3-0.4 mAP, not significant, p = 0.23-0.33). Naive concat and cross-attention are significantly worse, and late fusion collapses Boundary-F1 (0.09). Not NTU, no synthetic graded corruptions.
- **Relation to M4:** Peer-reviewed evidence that **per-frame skeleton reliability** helps avoid negative transfer under occlusion. It is the temporal-reliability precedent for M4's 8-segment reliability, but uses a heuristic pose-estimator confidence instead of a learned head, and it is a localization task.
- **Method weaknesses** (empirical; read: full text, partial):
  - Checked, found: significance tests.
  - [AA] Gains over RGB-only are not significant. The authors frame the method as a "safety net".

**24. Akgul, H., Eplik, M., Rojas, J., Yamamoto, A., Kumar, R., & Singh, M. (2025).** *RCMCL: A unified contrastive learning framework for robust multi-modal (RGB-D, skeleton, point cloud) action understanding* [Preprint]. arXiv. https://arxiv.org/abs/2511.04351 **[FLAGGED: unreviewed; author credibility unverified]**
- **Read scope:** abstract; Sec. III (AMG gate, Eq. 5); robustness section.
- **WHY:** Multimodal HAR degrades under sensor failure and noise.
- **HOW:**
  - Self-supervised cross-modal contrastive pre-training plus a degradation-simulation loss (skeleton denoising).
  - An **Adaptive Modality Gating network outputs a per-modality "reliability score" G_M ∈ [0, 1]** from features. The fused feature is the normalised G-weighted sum.
  - The gate is learned from the task loss. There is **no correctness target**. A per-frame gate visualisation is shown.
- **WHAT:** NTU RGB+D 120 and UWA3D-II. Modality dropout (RGB-D, skeleton missing). Skeleton Gaussian joint noise σ ∈ {0.05, 0.10, 0.15}. Point-cloud sparsity. Claims 11.5 % degradation under dual-modality dropout. No calibration metrics in the abstract (full results not checked for ECE).
- **Relation to M4:** If credible, it is the nearest "NTU-120 + skeleton noise/dropout + reliability gate" work. M4 differs by RGB (not RGB-D), supervised reliability (p(y_true)), calibration, selective prediction, and frozen supervised branches. **Do not cite it as established evidence** (see flag in Section 2).
- **Method weaknesses** (empirical; read: Sec. III + robustness):
  - Checked, none found: seeds/variance in the sections read.
  - The gate is trained without any reliability label. "Reliability score" is a naming claim rather than a validated estimate. [RI]

**25. Guo, M. (2026, August 11).** Lightweight probability-level fusion for occlusion-robust human activity recognition from RGB and skeleton data. *National High School Journal of Science*. https://nhsjs.com/2026/lightweight-probability-level-fusion-for-occlusion-robust-human-activity-recognition-from-rgb-and-skeleton-data/ **[FLAGGED: student journal; peer review of this item unconfirmed]**
- **Read scope:** web page via automated reading.
- **WHY:** Is cheap late fusion of RGB and skeleton experts robust to occlusion?
- **HOW:**
  - Pretrained C3D (RGB) and PoseC3D (skeleton) experts.
  - Fusion: probability averaging, log-probability averaging, and learned linear heads over probabilities / log-probabilities.
  - Synthetic occlusion: a fixed rectangular RGB mask and independent keypoint dropout, p ∈ {0, 10, 25, 40, 50, 70} %. **No uncertainty weighting.**
- **WHAT:** UCF-101 split 1 only.
  - Clean: log-prob averaging 94.08 %.
  - Train-clean/test-occluded: log-prob averaging best up to 50 % occlusion.
  - Matched-occlusion training lets the learned log-prob head reach 79.1 % at 40 %.
  - Uncertainty-aware fusion and NTU are listed as future work.
- **Relation to M4:** Its baseline set (probability averaging + learned heads over log-probabilities, clean vs occlusion-matched training) **closely mirrors M3 and M4's clean-trained vs corruption-aware comparison**, on a smaller dataset. Acknowledge it to pre-empt a novelty challenge. M4's uncertainty layer, NTU-120, six corruption families and calibration/AURC are all absent from it.
- **Method weaknesses** (empirical; read: page):
  - [AA] One dataset split, one occlusion seed, synthetic occlusion only.

---

## 4. Explicit gap flag (requested)

**Question:** Does any work already do uncertainty- or reliability-weighted RGB+skeleton fusion on NTU under synthetic corruptions?

**Answer:** **No verified peer-reviewed work found.** Bounded by the 10 web queries plus the OpenAlex/arXiv queries in Section 1 and the full-text checks of #21-#25.

| Work | RGB+skel | NTU | Synthetic graded corruptions | Reliability weight | Supervised reliability target | Calibration / selective prediction |
|---|---|---|---|---|---|---|
| AMFI-Net (#21) | yes | NTU-60 | no | learned attention | no | no |
| MAGNet (#22) | RGB+depth+pose | 60/120 | missing modality / joints only | "quality score" (unspecified) | no | no |
| Takami & Fukuda (#23) | yes | no | no (natural occlusion) | OpenPose confidence, per frame | no (heuristic) | no |
| RCMCL (#24, flagged) | RGB-D+skel+PC | NTU-120 | dropout + skeleton noise (3 levels) | learned gate | no | not in abstract |
| Guo (#25, flagged) | yes | no (UCF-101) | occlusion, 6 levels | fixed / linear heads | no | no |
| **M4 (planned)** | yes | NTU-120 | 6 families × 5 severities | entropy + learned head, per segment | **p(y_true)** | **ECE/NLL/Brier, AUROC, AURC** |

The reliability *estimators* M4 uses are established (TCP, MMDynamics, PDF, QMF). What is new is the **combination**: frozen video + skeleton branches, per-segment reliability, graded RGB/skeleton corruption families, calibration and selective prediction, and the clean vs corruption-aware training comparison.

---

## 5. Screened and excluded after full-text / metadata check

| Item | Reason |
|---|---|
| Dong, Y., Shi, Z., Li, X., Zhou, R.-G., & Ge, S. S. (2025). Multiview uncertainty-aware fusion for human activity recognition via Dempster-Shafer theory. *IEEE TII, 21*, 3037-3046. doi:10.1109/TII.2024.3523536 | Verified to exist, but it is **wearable-sensor HAR** (PAMAP2, MHEALTH, OPPORTUNITY, UCI-HAR). A web-search summary had wrongly described it as RGB/skeleton on NTU. Excluded as off-modality. Relevant only as evidence of DST fusion in HAR. |
| Zheng, X., et al. (2023). Multi-level confidence learning for trustworthy multimodal classification. *AAAI 37*(9), 11381-11389. doi:10.1609/aaai.v37i9.26346 | Verified. Multi-omics GCN; duplicates MMDynamics' framing. Excluded for scope. |
| Xiong, B., et al. (2024). Modality-collaborative test-time adaptation for action recognition. CVPR 2024. doi:10.1109/CVPR52733.2024.02524 | Verified. Prototype-based TTA for video domain shift, not reliability-weighted fusion. Excluded (READ covers multimodal TTA). |
| Lee, Y.-L., et al. (2023). Multimodal prompting with missing modalities for visual recognition. CVPR 2023. doi:10.1109/CVPR52729.2023.01435 | Verified (OpenAlex). Missing-modality prompting for vision-language. Ma et al. (2022) covers the point. |
| Bao, W., Yu, Q., & Kong, Y. (2021). Evidential deep learning for open set action recognition. ICCV 2021. doi:10.1109/ICCV48922.2021.01310 | Verified (OpenAlex). Unimodal evidential uncertainty; belongs to the unimodal-uncertainty theme. |
| Li & Cheng (2026), AGF-Net, arXiv 2610.30651; EMT-DS, arXiv 2610.13013 | Emotion-recognition preprints. Not RGB+skeleton; unreviewed. |
| Skeleton-only uncertainty (e.g. "uncertainty-guided spatial-temporal mutuality", Applied Soft Computing 2025; trust-gate ST-LSTM, arXiv 1706.08276) | Unimodal skeleton; belongs to another theme. |

## 6. UNVERIFIED (not annotated; gray zone = FAIL)

- **Bian, X., et al. (2025). Multimodal feature collaboration and fusion…** BMVC 2025 (bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_284). Seen only as a search hit. Title, authors and content not confirmed.
- **"Integrating spatio-temporal modeling of RGB video with multi-stream skeleton representations for advanced human action recognition."** *Neurocomputing* (ScienceDirect S0925231225024634). Seen only as a search hit. Authors, year and whether it uses adaptive or reliability weighting not confirmed (ScienceDirect blocked automated fetch).
- **"Skeleton Sequence and RGB Frame Based Multi-Modality Feature Fusion Network for Action Recognition"** (arXiv 2202.11374). Seen as a search hit only. Not fetched; appears to be static fusion.

## 7. Search and verification limitations

- Semantic Scholar was unavailable (HTTP 429), so there was no S2-ID deduplication. Deduplication was done by DOI/title through OpenAlex.
- OpenReview and ScienceDirect blocked automated fetches. The ICLR 2021 TMC venue was confirmed through mlanthology, the ICLR-hosted OpenReview PDF listing, the arXiv comment and the authors' repository citation, not through the forum page.
- Several entries are abstract-only (#3, #4, #10, #15-#20). Their method weaknesses are not assessed, per the rules.
- IEEE Xplore page ranges differ from CVF open-access ranges for CVPR papers. IEEE (DOI) pages are cited, with CVF pages noted where fetched.
- Google Scholar was not searched, so recent workshop papers on RGB+pose reliability may be missed. The gap claim in Section 4 is bounded accordingly.
- Wolpert (1992) is cited from metadata only. Its annotation reflects the standard description of stacking, not a re-reading.
