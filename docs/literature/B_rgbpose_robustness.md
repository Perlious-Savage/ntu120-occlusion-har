# Annotated Bibliography B: RGB-Pose Fusion and Occlusion/Corruption Robustness

**ARS lit-review, Phase 2 (bibliography + source verification).** Project: *Uncertainty-aware occlusion-robust RGB-skeleton action recognition on NTU RGB+D 120*. Frozen R3D-18 (RGB) + CTR-GCN (joint) branches. Milestone 3 benchmarked 6 synthetic corruptions x 5 severities plus dual corruption. Milestone 4 plans uncertainty/reliability-weighted fusion and corruption-aware training.

This file does only Phase 2 work: search, screening, verification and annotation. It contains no synthesis or draft text. The "Relation to project" lines are relevance notes, as requested.

---

## Headline positioning facts (verified)

1. **No verified prior work benchmarks RGB + skeleton fusion on NTU-120 under graded synthetic corruption of either or both modalities.** None of the searches below found a paper that combines (a) NTU RGB+D 120, (b) an RGB stream plus a skeleton stream, and (c) a severity grid of corruptions applied to RGB, to skeleton and to both. The nearest works each miss at least one of those three:
   - **AMR, Liu C. et al. 2024.** RGB + skeleton under attack on RGB only, skeleton only and both. The attacks are adversarial (FGSM, PGD, CW), not natural corruptions, and the dataset is NTU-**60**.
   - **ActionMAE, Woo et al. 2023.** Missing-modality tests on NTU-60/120, but the modalities are RGB, depth and IR. Skeleton is not used, and a modality is either present or absent, with no grading.
   - **Guo 2026.** Frozen RGB (C3D) + skeleton (PoseC3D) branches, both modalities occluded at 0-70%, several fusion rules compared. This is very close to our design, but it runs on **UCF-101** and appears in a high-school journal (grey literature).
   - **RCMCL, 2025 preprint.** Dual-modality dropout on NTU-120. Excluded, see below.
2. **Our skeleton corruptions follow the RA-GCN family of protocols.** Song et al. (2019; 2021) zero out joints on NTU-60/120 using five part groups (left arm, right arm, both hands, both legs, torso), frame occlusion, random joints (p = 0.2-0.6) and Gaussian jitter. Tanaka et al. (2024) and Parts-Mamba (2026) reuse these protocols. SHARDeg (2025) adds frame-rate and frame-block degradation on NTU-120 X-Sub/X-Set. All of this work is **skeleton-only**.
3. **The video-corruption benchmarks do not include actor-box occlusion.** These are ImageNet-C, Mini Kinetics-C / SSV2-C and Kinetics-P / UCF101-P. Their corruptions are noise, blur, weather, digital/compression and temporal (sampling, jumbling, freezing, packet loss). None of them is a black person-box occlusion, and none uses NTU. Both video benchmarks did evaluate 3D ResNets (R3D / 3D ResNet-18), close to our R3D-18.
4. **Kinect "inferred" joints are real and machine-readable in NTU.**
   - The official NTU parser (`read_skeleton_file.m`) reads one `trackingState` per joint per frame.
   - In the Kinect v2 SDK, 0 = NotTracked, 1 = Inferred, 2 = Tracked, and an inferred joint is one "that cannot be seen by the camera".
   - **No paper was found that uses NTU's per-joint trackingState as a natural-occlusion evaluation split or as a fusion reliability signal.** The nearest is ST-LSTM Trust Gates (Liu et al., 2018), which learns input reliability. The authors motivate it with Kinect noise and occlusion on NTU, but it does not use the flag.
5. **Robustness evidence that supports reliability-weighted fusion:**
   - AMR found that undefended RGB-skeleton fusion is **less robust than the skeleton branch alone** when one modality is attacked. Their learned modality reweighting improved this.
   - ActionMAE found that concatenation fusion collapses when a modality is removed.
   - PoseC3D reports that GCNs (MS-G3D) lose about 14 points under keypoint dropping, versus under 1 point for PoseConv3D.
6. **Correction to the brief: MMNet's first author is Bruce X. B. *Yu*, not "Bruce".** The DOI (Crossref) gives TPAMI vol. 45, issue 3 (2023), pp. 3522-3538. The 2022 date is the early-access year.

---

## Search Strategy

| Item | Value |
|---|---|
| **Databases / tools** | WebSearch (general web, standard and extended modes); arXiv export API (metadata for 20+ IDs); Crossref REST API (`query.bibliographic`, DOI metadata for 30+ works); OpenAlex (one lookup); CVF Open Access, NeurIPS D&B proceedings and ar5iv/arXiv HTML full text via WebFetch; Microsoft Learn (Kinect SDK docs); GitHub raw (NTU parser). The Semantic Scholar API returned HTTP 429, so Tier 0 was skipped (`[S2-API-UNAVAILABLE]`) and verification used Crossref/arXiv (Tier 1) and WebSearch (Tier 2) for every source instead of a 50% sample. |
| **Keywords** | "SHARDeg skeleton action recognition degradation benchmark"; "occluded skeleton action recognition NTU RGB+D robustness missing joints"; "RGB skeleton fusion action recognition missing modality robustness NTU RGB+D 120"; "skeleton action recognition robustness benchmark noise occlusion corruption GCN"; "Kinect joint tracking state inferred joints occlusion skeleton action recognition NTU"; "NTU RGB+D skeleton files trackingState"; "tracking state Kinect joints confidence weighting skeleton action recognition NTU"; "RGB pose fusion action recognition NTU RGB+D 120 robustness occlusion corrupted video modality 2024 2025"; "skeleton action recognition occlusion robust NTU 120 body part occlusion CTR-GCN 2024"; "multimodal action recognition RGB skeleton test-time corruption missing modality benchmark NTU-120 uncertainty reliability fusion"; "multimodal action recognition robustness benchmark RGB and skeleton both corrupted occlusion noise NTU late fusion"; exact-title searches for Schiappa 2023, Yi 2021, Peng 2023, AMR 2024, FineTec, PeGCN, MMNet, Hu 2024. |
| **Boolean logic** | (RGB OR video) AND (skeleton OR pose) AND (fusion OR multimodal) AND (robust* OR occlu* OR corrupt* OR missing); (skeleton) AND (occlu* OR noise OR missing joints OR degrad*) AND (NTU); (video action recognition) AND (corruption OR perturbation) AND benchmark |
| **Date range** | 2016-2026 for primary work. Seminal datasets/backbones (2016-2018) and ImageNet-C (2019) are kept as foundational. The brief asked for 2023-2026 coverage of RGB + skeleton under missing or corrupted input, and that period was searched. |
| **Last searched** | 2026-11-08 |
| **Language / types** | English; conference and journal papers, arXiv preprints (flagged), technical documentation (Kinect SDK, NTU repo) |
| **Inclusion criteria** | (a) RGB + pose/skeleton fusion for action recognition, preferably reporting NTU-60/120; or (b) skeleton action recognition under occlusion, noise, missing joints/frames or adversarial perturbation; or (c) corruption/perturbation benchmarks for video action recognition; or (d) the datasets/backbones used by the project. Existence and metadata confirmed via DOI/Crossref, arXiv or the proceedings page. |
| **Exclusion criteria** | Existence or metadata unconfirmable (gray zone = FAIL); content inaccessible beyond metadata (listed, not annotated); tangential (zero-shot, multi-entity debiasing, RGB-D without skeleton robustness, clean-only fusion); unreviewed preprints whose claims could not be checked against a reviewed version. |
| **Coverage distribution advisory** | `DISTRIBUTIONAL_SKEW_ADVISORY`: **Methodological distribution.** Computational/empirical (method or benchmark) = 26/28 (93%); dataset/tool papers and documentation make up the rest. Advisory: this is a coverage signal, not a defect, and is expected for an ML engineering topic. Search response: no expansion; the scope was set by the user. `DISTRIBUTIONAL_SKEW_ADVISORY`: **Venue tier distribution.** Top CV/ML venues (CVPR/ICCV/ECCV/TPAMI/NeurIPS/ICLR/AAAI) = 15/28 (54%); below the 70% threshold, so nothing is emitted. Time: 2016-2020 = 8, 2021-2023 = 9, 2024-2026 = 11, so no skew. |

### PRISMA-style flow (approximate; web search does not give exact hit counts)

```
Records identified (total): ~175
|-- WebSearch result links (17 queries x ~10): ~170
|-- Other (known-item lookups via arXiv/Crossref, NTU repo, Kinect docs): ~5
Duplicates / same-paper mirrors removed: ~110
Unique records screened (title/abstract): ~65
Records excluded at screening (off-topic, surveys, non-action, mirrors): ~28
Full text / abstract assessed: 37
Excluded after assessment (reasons below): 9
Included and annotated: 28
```

**Excluded after assessment (existence verified, not annotated):**
- **RCMCL** (Akgul et al., arXiv:2511.04351, 2025). Claims dual-modality-dropout robustness on NTU-120 (RGB-D, skeleton, point cloud). Excluded because it is an unreviewed preprint with no reviewed version found, its 11.5% degradation figure could not be checked against tables in this pass, and its authors' affiliations could not be confirmed. It is relevant enough that the caller may wish to check it by hand.
- **Hu, Z., Xiao, J., Li, L., Liu, C., & Ji, G. (2024).** Human-centric multimodal fusion network for robust action recognition. *Expert Systems with Applications, 239*, 122314. https://doi.org/10.1016/j.eswa.2023.122314. Metadata verified via Crossref and OpenAlex. The abstract and full text were inaccessible (ScienceDirect 403, no abstract in OpenAlex), so it is not annotated. **It may be directly relevant (RGB + human-centric fusion, "robust"); check by hand.**
- **CHASE** (Wen et al., NeurIPS 2024). Tangential: about multi-entity debiasing. It does report CTR-GCN under masking and noise.
- **Visual Anchoring in Diffusion** (Bao, Guo & Yu, arXiv:2608.04623, 2026). Zero-shot setting.
- **PAN** (Liang et al., arXiv:2512.21916, 2025). Clean RGB-skeleton fusion only.
- **EPP-Net** (human parsing + pose). No corruption tests.
- **MAGNet** (Sci. Rep. 2026). No quantitative occlusion results in the accessible text.
- **Heatmap Pooling Network** (arXiv:2512.03837). No corruption tests.
- **VPN++** (Das et al., TPAMI 2022, doi:10.1109/TPAMI.2021.3127885). Verified. It extends VPN and is mentioned under VPN rather than given a separate entry.

### UNVERIFIED (gray zone = FAIL; not annotated)
- None of the requested sources failed verification. Every item in the brief was confirmed with metadata, with one **metadata correction**: MMNet's authorship and year (see Fact 6).
- One **claim** was rejected rather than a source. A search snippet said ActionMAE's NTU-60 accuracy falls "93.3% → 59.4% when RGB removed". This did not match the paper's full text, which gives sum fusion 93.3 → 89.2 and concat 93.2 → 18.6 when RGB is dropped. Only the full-text figures are used below.

---

## Sources (N = 28)

Evidence-level note: the 7-level hierarchy in the role file is clinical. For ML papers it is mapped as follows: controlled computational experiments (methods and benchmarks) = **Level III** (controlled, non-randomized); dataset papers = **Level VI** (descriptive); vendor documentation = **Level VII**. COI: no financial COI was identified for any source. Intellectual COI (authors benchmarking their own method) applies to all method papers and is not repeated per entry. No predatory venues were found. NHSJS (Guo 2026) is a student journal and is flagged under its entry.

Read-scope legend:
- **FT-HTML**: full text read through arXiv/ar5iv HTML (WebFetch extraction).
- **ABS**: abstract or metadata only.
- **DOC**: documentation or source code.

### Theme 4: Datasets and backbones

**1. Shahroudy, A., Liu, J., Ng, T.-T., & Wang, G. (2016). NTU RGB+D: A large scale dataset for 3D human activity analysis. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1010-1019). https://doi.org/10.1109/CVPR.2016.115**
- Verification: VERIFIED (Crossref DOI + arXiv:1604.02808). Level VI. Read: FT-HTML.
- **WHY:** Earlier RGB-D/skeleton benchmarks were too small, with too few classes, subjects and views.
- **HOW:** Three Kinect v2 cameras recorded 56,880 samples: 60 classes, 40 subjects, 80 viewpoints; RGB 1920x1080, depth/IR 512x424, 25 3D joints.
  - Cross-Subject (X-Sub): 20 training subjects; 40,320 train / 16,560 test.
  - Cross-View: cameras 2 and 3 for training, camera 1 for test.
- **WHAT:** Skeleton methods generalise across views but are "prone to errors of the body tracker". The authors filter false body detections (e.g., seats and tables detected as bodies) with an X/Y-spread rule.
- **Method weaknesses** (dataset paper; read FT-HTML): occlusion and the tracking-state flags are not analysed (checked: dataset description and protocol sections). [reader-inferred]
- **Relation to project:** The parent dataset. It confirms that Kinect skeleton noise and tracking error exist in the raw data, which supports a natural-occlusion analysis.

**2. Liu, J., Shahroudy, A., Perez, M., Wang, G., Duan, L.-Y., & Kot, A. C. (2020). NTU RGB+D 120: A large-scale benchmark for 3D human activity understanding. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 42*(10), 2684-2701. https://doi.org/10.1109/TPAMI.2019.2916873**
- Verification: VERIFIED (Crossref + arXiv:1905.04757). Level VI. Read: FT-HTML (first ~95%).
- **WHY:** Scale up NTU-60 and add a one-shot protocol.
- **HOW:** 114,480 samples, 120 classes (82 daily, 12 health, 26 mutual), 106 subjects, 32 setups, 155 viewpoints, Kinect v2.
  - X-Sub: 53/53 subjects.
  - X-Set: even setup IDs for training, odd for test.
- **WHAT:**
  - The tracker output "is not perfect and can be noisy sometimes".
  - Modality baselines (X-Sub / X-Set): RGB 58.5 / 54.8; skeleton 55.7 / 57.9; RGB + skeleton 61.2 / 63.1.
  - Skeleton generalises better across setups.
  - The NTU repo lists 535 NTU-120 samples with missing or incomplete skeletons to be ignored (verified on github.com/shahroudy/NTURGB-D).
- **Method weaknesses** (dataset paper): no analysis of tracking-state flags or occlusion frequency (checked: dataset and limitations discussion). [reader-inferred]
- **Relation to project:** The project's benchmark and both protocols. The 535 missing-skeleton samples matter for paired RGB-skeleton evaluation; confirm they are handled consistently in both branches.

**3. Chen, Y., Zhang, Z., Yuan, C., Li, B., Deng, Y., & Hu, W. (2021). Channel-wise topology refinement graph convolution for skeleton-based action recognition. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 13339-13348). https://doi.org/10.1109/ICCV48922.2021.01311**
- Verification: VERIFIED (Crossref + arXiv:2107.12213). Level III. Read: ABS + metadata.
- **WHY / HOW / WHAT:** CTR-GCN learns a shared topology prior and refines it per channel. Robustness is not studied. (Third-party evidence of CTR-GCN robustness: under FineTec's temporal corruption on NTU-120 X-Sub, it scores 81.4 / 80.9 / 79.3 at 25 / 50 / 75% frame drop.)
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** The skeleton branch. The joint stream only is used, frozen.

**4. Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). A closer look at spatiotemporal convolutions for action recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 6450-6459). https://doi.org/10.1109/CVPR.2018.00675**
- Verification: VERIFIED (Crossref + arXiv:1711.11248). Level III. Read: ABS + metadata.
- **WHY / HOW / WHAT:** Compares 2D, mixed, full 3D (R3D) and (2+1)D residual convolutions. It is the source of the R3D-18 / R(2+1)D families in torchvision.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** The RGB backbone (R3D-18, Kinetics-pretrained). Schiappa et al. (2023) and Yi et al. (2021) both report corruption robustness for 3D ResNets, so we have external reference points.

**5. Microsoft. (2014). *TrackingState enumeration* [Kinect for Windows SDK 2.0 API reference; archived]. Microsoft Learn. https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn758896(v=ieb.10)**
- Additional source: Shahroudy, A. (n.d.). *NTURGB-D* [Code repository; `Matlab/read_skeleton_file.m`]. GitHub. https://github.com/shahroudy/NTURGB-D
- Verification: VERIFIED (page fetched; parser source fetched). Level VII. Read: DOC (full).
- **WHAT:**
  - Enum values: NotTracked = 0, Inferred = 1, Tracked = 2.
  - Inferred means "confidence in the position data is lower than if it were Tracked". "A tracked joint that cannot be seen by the camera is inferred", meaning its position is calculated from surrounding joints.
  - The NTU parser reads, per joint, 11 floats (3D xyz, depth xy, colour xy, orientation wxyz) followed by an integer `trackingState`.
  - The NTU README confirms the data come from Kinect v2.
- **Relation to project:** This is the citable basis for the reviewers' "natural occlusion" point. Per-joint inferred flags allow (a) a natural-occlusion test subset, such as samples above an inferred-joint fraction, and (b) a reliability signal for fusion.
  - No published action-recognition paper was found that uses these flags this way (searches 5-7 above). This is a gap, stated with a bounded search, not proof of absence.

### Theme 1: RGB + pose/skeleton fusion

**6. Das, S., Sharma, S., Dai, R., Brémond, F., & Thonnat, M. (2020). VPN: Learning video-pose embedding for activities of daily living. In *Computer Vision - ECCV 2020* (Lecture Notes in Computer Science, pp. 72-90). Springer. https://doi.org/10.1007/978-3-030-58545-7_5**
- Verification: VERIFIED (Crossref + arXiv:2007.03056). Level III. Read: FT-HTML.
- **WHY:** RGB 3D-CNNs miss fine-grained, pose-defined ADL distinctions.
- **HOW:** I3D (ImageNet + Kinetics) features are modulated by a pose-driven spatio-temporal attention network (a GCN plus a coupler). A spatial embedding loss aligns the visual and pose attention.
- **WHAT (NTU numbers):**
  - NTU-120: **86.3 X-Sub / 87.8 X-Set**.
  - NTU-60: 93.5 / 96.2 (95.5 / 98.0 with ResNeXt-101).
  - No robustness experiments. Failures are attributed to occlusion, low resolution and low-quality poses, and handling noisy poses is listed as future work.
  - The extension VPN++ (TPAMI 2022) is verified but was not read.
- **Method weaknesses** (empirical; read FT-HTML):
  - checked, found: external validity (4 datasets).
  - checked, none found: robustness / internal-validity tests on pose quality.
  - not checked: statistical reporting.
  - The pose source per dataset is ambiguous (Kinect vs LCRNet), so cross-dataset gains could partly reflect pose quality. [author text ambiguous; reader-inferred]
- **Relation to project:** A canonical pose-guided RGB fusion baseline on NTU-120. Our clean fusion (about 89.5-89.9 X-Sub, about 92 X-Set) sits above VPN. VPN assumes reliable pose, and our corruption benchmark tests exactly that assumption.

**7. Duan, H., Zhao, Y., Chen, K., Lin, D., & Dai, B. (2022). Revisiting skeleton-based action recognition. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2959-2968). https://doi.org/10.1109/CVPR52688.2022.00298**
- Verification: VERIFIED (Crossref + arXiv:2104.13586, "CVPR 2022 Oral"). Level III. Read: FT-HTML (ar5iv version of the arXiv paper).
- **WHY:** GCNs are brittle to pose noise and hard to fuse with RGB.
- **HOW:**
  - PoseConv3D stacks 2D keypoint heatmaps (HRNet-estimated, not Kinect) into a 3D-CNN input.
  - RGBPoseConv3D has two pathways with bidirectional lateral connections (after res3/res4) plus late fusion.
- **WHAT:**
  - RGBPoseConv3D NTU-120: **95.3 X-Sub / 96.4 X-Set**; NTU-60: 97.0 / 99.6.
  - PoseConv3D joint-only NTU-120: 86.0 / 89.6.
  - Keypoint dropping (one limb keypoint per frame with probability p = 1): PoseConv3D loses under 1 point, MS-G3D loses 14.3 points.
  - Estimated 2D poses beat Kinect 3D poses on NTU-60 (91.9 vs 89.4).
- **Method weaknesses** (empirical; read FT-HTML):
  - checked, found: operational definitions, external validity.
  - checked, none found: corruption of the RGB pathway, robustness of the fused model.
  - not checked: variance across seeds.
  - The robustness test perturbs only the pose input of pose-only models. The fused RGBPose model is never stressed. [reader-inferred]
- **Relation to project:** The strongest RGB-pose fusion SOTA on NTU-120; we should position as robustness analysis rather than competing on clean accuracy. The keypoint-drop result is the closest prior evidence on GCN fragility to joint dropout. We extend it to RGB, to dual corruption and to a frozen CTR-GCN with Kinect 3D joints.

**8. Yu, B. X. B., Liu, Y., Zhang, X., Zhong, S.-h., & Chan, K. C. C. (2023). MMNet: A model-based multimodal network for human action recognition in RGB-D videos. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 45*(3), 3522-3538. https://doi.org/10.1109/TPAMI.2022.3177813**
- Verification: VERIFIED (Crossref). **Metadata correction:** the first author is Bruce X. B. **Yu**; published 2023 (early access 2022). Level III. Read: ABS (the PDF was not machine-readable).
- **WHY / HOW:** A skeleton ST-GCN learns attention weights that are transferred to the RGB network: model-level fusion of skeleton and RGB.
- **WHAT:**
  - Evaluated on NTU-60, NTU-120, PKU-MMD, NW-UCLA and Toyota Smarthome.
  - NTU-120 **92.9 X-Sub / 94.4 X-Set**; NTU-60 96.0 / 98.8. These figures come from Papers-with-Code and a Sensors (2024) comparison table, not from the primary PDF; treat them as secondary.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** Skeleton-guided RGB attention ties RGB quality to skeleton quality. Under our skeleton corruptions, such coupling could spread the damage to the RGB stream; our late fusion of frozen branches avoids this by design. Not tested for robustness.

**9. Reilly, D., & Das, S. (2024). Just add π! Pose induced video transformers for understanding activities of daily living. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 18340-18350). https://doi.org/10.1109/CVPR52733.2024.01736**
- Verification: VERIFIED (Crossref + arXiv:2311.18840). Level III. Read: FT-HTML.
- **WHY:** Use pose during training without needing it at inference.
- **HOW:** 2D and 3D skeleton-induction auxiliary modules on TimeSformer are removed at test time.
- **WHAT:**
  - RGB-only NTU-120: **91.9 X-Sub / 92.9 X-Set**.
  - An RGB + 3D-pose late-fusion variant (with Hyperformer) reaches 95.1 / 96.1.
  - Appendix: 2D-SIM degrades to baseline under heavy pose noise; 3D-SIM stays above baseline at all noise levels tested.
- **Method weaknesses** (empirical; read FT-HTML):
  - checked, found: external validity (Smarthome, NTU-60/120).
  - checked, found: a pose-noise sensitivity study (Appendix C).
  - not checked: seed variance.
  - The noise tests perturb training-time pose only, so test-time robustness of the fused variant is not measured. [reader-inferred]
- **Relation to project:** An alternative design that removes test-time pose dependence. A useful contrast for the "what if the skeleton is unreliable at test time" argument.

**10. Woo, S., Lee, S., Park, Y., Nugroho, M. A., & Kim, C. (2023). Towards good practices for missing modality robust action recognition. *Proceedings of the AAAI Conference on Artificial Intelligence, 37*(3), 2776-2784. https://doi.org/10.1609/aaai.v37i3.25378**
- Verification: VERIFIED (Crossref + arXiv:2211.13916, "AAAI 2023 Oral"). Level III. Read: FT-HTML.
- **WHY:** Multimodal action models fail when a modality is absent at test time.
- **HOW:** Train with RGB + depth + IR, then drop modalities at inference.
  - Fusions compared: sum, concat, transformer.
  - ActionMAE randomly drops modality tokens during training and reconstructs the missing features.
- **WHAT (NTU-60, RGB dropped):**
  - Sum 93.3 → 89.2; concat 93.2 → 18.6; transformer 92.6 → 90.4.
  - ActionMAE trained on RGB + depth, NTU-120 X-Sub: 91.5 complete / 84.7 RGB-only / 87.0 depth-only.
  - Skeleton is not used by the method.
- **Method weaknesses** (empirical; read FT-HTML):
  - checked, found: operational definitions, external validity (4 datasets).
  - checked, none found: partial or graded corruption; only full presence or absence.
  - not checked: seeds.
  - The binary missing-modality protocol does not cover partial degradation, where a modality is present but unreliable. [reader-inferred]
- **Relation to project:** The closest missing-modality precedent on NTU-120, but with different modalities and binary removal. Our graded, both-modality corruption and our "corruption-aware training" in M4 parallel its modality-dropout training.

**11. Liu, C., Liu, X., Yu, Z., Hou, Y., Yue, H., & Yang, J. (2024). Adversarial robustness in RGB-skeleton action recognition: Leveraging attention modality reweighter. In *2024 IEEE International Joint Conference on Biometrics (IJCB)* (pp. 1-11). https://doi.org/10.1109/IJCB62174.2024.10744468**
- Verification: VERIFIED (Crossref + arXiv:2407.19981). Level III. Read: FT-HTML.
- **WHY:** The first study of adversarial robustness for RGB-skeleton fusion.
- **HOW:** I3D (RGB) + HCN (skeleton), on NTU-60 X-Sub and iMiGUE. FGSM, PGD20 and CW attacks target RGB only, skeleton only, or both. The Attention-based Modality Reweighter (AMR) learns modality weights and adds auxiliary losses.
- **WHAT:**
  - Undefended fusion on NTU-60: clean 88.09; PGD20 accuracy 2.09 with RGB attacked, 11.84 with skeleton attacked, 0.15 with both.
  - Skeleton is more robust than RGB.
  - Fusion can be **less** robust than the skeleton alone.
  - AMR under adversarial training: clean 81.65, both attacked 66.52.
  - NTU-120 not used.
- **Method weaknesses** (empirical; read FT-HTML):
  - checked, found: operational definitions of attack settings.
  - checked, none found: natural corruptions, NTU-120.
  - not checked: seeds.
  - Weak backbones (HCN, I3D) were chosen deliberately; the findings may not hold for SOTA backbones. [author-acknowledged, section: experimental setup ("simpler models")]
- **Relation to project:** **The nearest precedent for per-modality and both-modality stress-testing of RGB-skeleton fusion, and for learned modality reweighting.** Our work differs in three ways: natural/synthetic corruptions rather than adversarial ones; NTU-120 X-Sub/X-Set rather than NTU-60; and uncertainty-guided rather than attention-learned weights.

**12. Guo, M. (2026, August 11). Lightweight probability-level fusion for occlusion-robust human activity recognition from RGB and skeleton data. *National High School Journal of Science (NHSJS)*. https://nhsjs.com/2026/lightweight-probability-level-fusion-for-occlusion-robust-human-activity-recognition-from-rgb-and-skeleton-data/**
- Verification: PLAUSIBLE (page fetched; no DOI). **Flag:** student journal; the peer review applied to this article could not be confirmed. Level III design, low venue tier.
- Read: FT (web page).
- **WHY / HOW:**
  - Frozen C3D (RGB) + frozen PoseC3D (skeleton) on UCF-101 split 1.
  - RGB occlusion: one fixed erased rectangle per clip.
  - Skeleton occlusion: independent keypoint dropout with probability p.
  - Both occluded at 0-70%.
  - Fusion rules: average probability, average log-probability, learned linear heads. Protocols: train-clean/test-occluded and matched occlusion.
- **WHAT:**
  - Clean: RGB 83.08, skeleton 86.81, log-prob fusion 94.08.
  - At 40% occlusion (train-clean): log-prob fusion 74.12 vs skeleton 62.38 vs RGB 36.82.
  - Learned log-prob fusion is best when trained on matched occlusion.
- **Method weaknesses:** single seed, single dataset, synthetic occlusion. [author-acknowledged, section: limitations]
- **Relation to project:** Methodologically the most similar design found (frozen branches, both modalities occluded, fusion-rule comparison). It differs from ours in dataset (UCF-101 vs NTU-120), occlusion types (random rectangle vs actor-box), absence of a severity-by-type grid, and absence of uncertainty weighting. Cite cautiously as grey literature.

**13. Babey, N., Gu, T., Li, Y., Meo, C., & Zhu, K. (2025). *Grounding foundational vision models with 3D human poses for robust action recognition* [Preprint; NeurIPS 2025 SpaVLE workshop]. arXiv. https://arxiv.org/abs/2511.05622**
- Verification: PLAUSIBLE (arXiv metadata; the workshop acceptance is stated in the arXiv comment only). Level III. Read: ABS.
- **WHY / HOW / WHAT:** Fuses V-JEPA 2 video features with occlusion-tolerant CoMotion poses. Evaluated on InHARD and the high-occlusion UCF-19-Y-OCC; the authors report the largest gains in occluded scenes.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** Evidence that pose helps RGB under **natural** occlusion, on non-NTU data. It supports the motivation, not the protocol.

### Theme 2: Skeleton robustness (occlusion, noise, missing data, adversarial)

**14. Liu, J., Shahroudy, A., Xu, D., Kot, A. C., & Wang, G. (2018). Skeleton-based action recognition using spatio-temporal LSTM network with trust gates. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 40*(12), 3007-3021. https://doi.org/10.1109/TPAMI.2017.2771306**
- Verification: VERIFIED (Crossref; the ECCV 2016 conference version is also on Crossref). Level III. Read: ABS + search snippet of full text.
- **WHY / HOW / WHAT:** A trust gate in the ST-LSTM learns the reliability of each input step, motivated by Kinect noise and "frequently occluded" joints in NTU, especially in side views. It reports clear gains on NTU.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** An early precedent for learned per-input reliability on NTU skeletons. It is conceptually related to M4 reliability weighting, but it works within one modality and does not use the Kinect trackingState.

**15. Song, Y.-F., Zhang, Z., & Wang, L. (2019). Richly activated graph convolutional network for action recognition with incomplete skeletons. In *2019 IEEE International Conference on Image Processing (ICIP)* (pp. 1-5). https://doi.org/10.1109/ICIP.2019.8802917**
- Verification: VERIFIED (Crossref + arXiv:1905.06774). Level III. Read: ABS.
- **WHY / HOW / WHAT:**
  - RA-GCN, conference version: multi-stream GCN in which class activation maps (CAM) pass the less-activated joints to later streams.
  - A synthetic NTU occlusion dataset, with joints occluded over space and time.
  - Performance stays comparable on clean data and degrades much less under occlusion.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** The origin of the NTU synthetic skeleton-occlusion protocol. See #16 for details.

**16. Song, Y.-F., Zhang, Z., Shan, C., & Wang, L. (2021). Richly activated graph convolutional network for robust skeleton-based action recognition. *IEEE Transactions on Circuits and Systems for Video Technology, 31*(5), 1915-1925. https://doi.org/10.1109/TCSVT.2020.3015051**
- Verification: VERIFIED (Crossref + arXiv:2008.03791). Level III. Read: FT-HTML.
- **WHY:** GCNs over-rely on a few informative joints, so occlusion and jitter hurt them badly.
- **HOW:** Train clean, test degraded. Corruptions are applied by zeroing joints before preprocessing, on NTU-60 X-Sub and NTU-120 **X-Set**:
  - **Frame occlusion:** 10-50 frames zeroed within the first 100.
  - **Part occlusion:** left arm, right arm, both hands, both legs, torso.
  - **Block occlusion:** joints below a height line.
  - **Random occlusion:** each joint with p = 0.2-0.6.
  - **Jitter:** Gaussian σ = 0.05 / 0.1 on random joints, with p = 0.02-0.10.
- **WHAT:**
  - Clean NTU-120: 3s RA-GCN 81.1 X-Sub / 82.7 X-Set.
  - NTU-120 X-Set, torso occluded: ST-GCN 48.5 vs RA-GCN 69.9.
  - NTU-60, 50 frames occluded: 24.0 vs 38.5.
  - NTU-60, jitter: 44.1 vs 61.6.
- **Method weaknesses** (empirical; read FT-HTML):
  - checked, found: operational definitions, external validity (two datasets).
  - checked, none found: seed or variance reporting.
  - not checked: hypothesis clarity.
  - Occlusion uses zeros in coordinate space, which models see as a real position at the origin rather than "missing", so results mix missingness effects with out-of-distribution input effects. [reader-inferred]
- **Relation to project:** **Our skeleton corruptions (random joint dropout, body-part dropout, Gaussian joint noise) closely mirror this protocol.** We should cite it as the protocol ancestor and say what differs:
  - our part groups (leg / arm / lower body / upper body) vs their five parts;
  - 5 severities;
  - evaluation on both NTU-120 X-Sub and X-Set (they used X-Set only for NTU-120);
  - a frozen CTR-GCN;
  - our main addition: RGB and dual corruption with fusion.

**17. Yoon, Y., Yu, J., & Jeon, M. (2022). Predictively encoded graph convolutional network for noise-robust skeleton-based action recognition. *Applied Intelligence, 52*(3), 2317-2331. https://doi.org/10.1007/s10489-021-02487-z**
- Verification: VERIFIED (Crossref; arXiv:2003.07514 preprint, which lists the authors in a different order). Level III. Read: ABS.
- **WHY / HOW / WHAT:** Captured skeletons may be incomplete or noisy. Predictive coding maximises the mutual information between clean and noisy skeleton representations. Evaluated on NTU-RGB+D and Kinetics-Skeleton, with gains reported on noisy samples.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** A skeleton-only noise-robust training precedent, relevant to M4 corruption-aware training.

**18. Wang, H., He, F., Peng, Z., Shao, T., Yang, Y.-L., Zhou, K., & Hogg, D. (2021). Understanding the robustness of skeleton-based action recognition under adversarial attack. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 14651-14660). https://doi.org/10.1109/CVPR46437.2021.01442**
- Verification: VERIFIED (Crossref + arXiv:2103.05347). Level III. Read: FT-HTML.
- **WHY / HOW:** The SMART attack, with perceptual constraints on bone length and motion dynamics. Tested on HDM05, MHAD and NTU-60 against ST-GCN, AS-GCN, DGNN, 2s-AGCN, MS-G3D, SGN and others.
- **WHAT:**
  - White-box untargeted success of about 98% on NTU.
  - 81.9% of user-study participants could not spot the perturbations.
  - Models are over-sensitive to high-velocity joints.
- **Method weaknesses** (empirical; read FT-HTML): checked, found: perceptual user study; not checked: seeds. Adversarial threat model only. [reader-inferred]
- **Relation to project:** Shows GCN fragility under worst-case perturbations. Our Gaussian joint noise is the natural, average-case counterpart.

**19. Peng, K., Roitberg, A., Yang, K., Zhang, J., & Stiefelhagen, R. (2023). Delving deep into one-shot skeleton-based action recognition with diverse occlusions. *IEEE Transactions on Multimedia, 25*, 1489-1504. https://doi.org/10.1109/TMM.2023.3235300**
- Verification: VERIFIED (Crossref + arXiv:2202.11423). Level III. Read: FT-HTML (first ~86%).
- **WHY:** Occlusion robustness in one-shot skeleton recognition.
- **HOW:**
  - Random occlusion: frames and joints zeroed.
  - **Realistic occlusion:** IKEA/PIX3D furniture meshes projected into each camera view, with joints inside the projected hull zeroed.
  - Uses the NTU-120 one-shot protocol (100 base / 20 novel classes), NTU-60 and Toyota Smarthome.
  - The Trans4SOAR transformer fuses joint, velocity and bone streams.
- **WHAT:**
  - NTU-120 one-shot: clean 57.05, realistic 52.35, random 53.17.
  - Realistic occlusion is generally harder than random.
- **Method weaknesses** (empirical; read FT-HTML): checked, found: operational definitions of occlusion. The one-shot protocol limits comparability with standard X-Sub/X-Set. [reader-inferred]
- **Relation to project:** A more physically grounded skeleton-occlusion generator (scene-object occluders). It suggests our body-part dropout may understate realistic occlusion difficulty. Different protocol (one-shot).

**20. Tanaka, N., Kera, H., & Kawamoto, K. (2024). Fourier analysis on robustness of graph convolutional neural networks for skeleton-based action recognition. *Computer Vision and Image Understanding, 240*, 103936. https://doi.org/10.1016/j.cviu.2024.103936**
- Verification: VERIFIED (Crossref + arXiv:2305.17939). Level III. Read: FT-HTML.
- **WHY / HOW:**
  - Fourier heatmaps of GCN sensitivity.
  - Corruptions on NTU-60 X-Sub: Gaussian noise (σ = 0.01-0.05), frame loss (p = 0.4-0.8), and part occlusion (RA-GCN's five parts).
  - Models: ST-GCN and TCA-GCN, standard vs adversarially trained.
- **WHAT:**
  - Adversarial training greatly improves robustness to Gaussian noise.
  - Standard training is more robust to part occlusion.
  - The authors state that part occlusion "cannot be explained using the Fourier heatmaps" because it is a missing-signal problem.
- **Method weaknesses** (empirical; read FT-HTML): checked, found: external validity limited to NTU-60. [reader-inferred]
- **Relation to project:** Directly relevant to M4 corruption-aware training. Training for noise may not transfer to occlusion; this fits our separate joint-noise and part-dropout axes.

**21. Liu, C., Jiang, Y., Du, C., & Li, Z. (2024). Enhancing action recognition from low-quality skeleton data via part-level knowledge distillation. *Signal Processing, 221*, 109486. https://doi.org/10.1016/j.sigpro.2024.109486**
- Verification: VERIFIED (Crossref + arXiv:2404.18206). Level III. Read: ABS.
- **WHY / HOW / WHAT:** Low-quality skeletons have missing or inaccurate joints. A teacher trained on high-quality skeletons guides a low-quality student, using part-based matching and a part-level contrastive loss. Evaluated on NTU RGB+D, Penn Action and SYSU 3D HOI.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** A distillation-based alternative to corruption-aware training. Skeleton only.

**22. Chen, Y., Peng, K., Roitberg, A., Schneider, D., Zhang, J., Zheng, J., Chen, Y., Liu, R., & Yang, K. (2025). Exploring self-supervised skeleton-based action recognition in occluded environments. In *2025 International Joint Conference on Neural Networks (IJCNN)* (pp. 1-9). https://doi.org/10.1109/IJCNN64981.2025.11228855**
- Verification: VERIFIED (Crossref + arXiv:2309.12029, "Accepted to IJCNN 2025"). Level III. Read: FT-HTML.
- **WHY / HOW:**
  - Realistic furniture-projection occlusion and random 20% coordinate occlusion on NTU-60 (xsub/xview) and NTU-120 (**xsub/xset**).
  - OPSTL: adaptive spatial masking plus a KNN imputer in latent space.
- **WHAT:**
  - Realistic occlusion, linear evaluation, NTU-120: 59.29 xsub / 58.25 xset (IosPSTL).
  - Random 20% occlusion collapses NTU-120 linear evaluation to 10.32 / 4.22. Imputation lifts this to 23.54 / 18.59.
- **Method weaknesses** (empirical; read FT-HTML): checked, none found: clean-accuracy baseline in the extracted text, so the size of the drop cannot be computed. [reader-inferred]
- **Relation to project:** NTU-120 X-Sub and X-Set occlusion evaluation, skeleton only and self-supervised. Imputation is an alternative to down-weighting a corrupted modality.

**23. Malzard, S., Mital, N., Walters, R., Nockles, V., Rao, R., & De Melo, C. M. (2025). *SHARDeg: A benchmark for skeletal human action recognition in degraded scenarios* (arXiv:2505.18048) [Preprint]. arXiv. https://arxiv.org/abs/2505.18048**
- Verification: PLAUSIBLE. It exists on arXiv (v1 23 May 2025, v2 27 May 2025). **No peer-reviewed version was found** (the Crossref search returned no match). Level III. Read: FT-HTML.
- **WHY:** The authors describe it as the first data-degradation benchmark on NTU-120 skeletons.
- **HOW:**
  - Test-time degradation on X-Sub and X-Set: random frame subsampling (0-90%), uniform subsampling (15/10/5/3 FPS), and single contiguous block dropout ("complete skeletal occlusion over many frames").
  - Models: MS-G3D, FR-HEAD, DeGCN, LogSigRNN, GCN-Dev-LSTM.
  - Mitigation: linear interpolation of the dropped frames.
- **WHAT:**
  - Clean X-Sub: DeGCN 87.2.
  - At equal effective frame rate, accuracy varies by more than 40 points across degradation types; irregular dropout is worst.
  - Interpolation recovers more than 40 points for MS-G3D.
  - LogSigRNN beats DeGCN under heavy degradation.
  - No RGB, and no joint-level dropout or noise (listed as future work).
- **Method weaknesses** (empirical; read FT-HTML): some cross-setup differences "could be within error bounds". [author-acknowledged, section: results]
- **Relation to project:** **Exists. It is the NTU-120 skeleton degradation benchmark the reviewers likely mean, but it covers frame-level skeleton degradation only.**
  - Our skeleton axes (joint/part dropout, noise) are complementary.
  - Our temporal frame mask on **RGB** parallels its block dropout on skeleton.
  - No CTR-GCN, no RGB, no fusion.

**24. Shen, T., Xu, H., Ahuja, N., Tickoo, O., Shin, P., & Narayanan, V. (2026). Parts-Mamba: Augmenting joint context with part-level scanning for occluded human skeleton. In *2026 IEEE International Conference on Image Processing (ICIP)* (pp. 1-6). https://doi.org/10.1109/ICIP61757.2026.11630168**
- Verification: VERIFIED (Crossref + arXiv:2511.16860). Level III. Read: FT-HTML.
- **WHY / HOW:**
  - A hybrid of a GCN (DeGCN) and Mamba, scanning at part level to recover context from distant joints.
  - Spatial occlusion uses RA-GCN's five parts. Temporal occlusion masks 0-50% of consecutive frames; results averaged over 5 runs.
  - Train clean, test occluded, on NTU-60/120.
- **WHAT:**
  - Mean accuracy over the spatial occlusion settings, NTU-120: 82.7 vs the best prior (MSFGCN) 73.4.
  - Temporal, NTU-120: 74.1 vs 3s RA-GCN 60.8.
  - Clean NTU-120: 90.2.
  - CTR-GCN is among the evaluated baselines.
- **Method weaknesses** (empirical; read FT-HTML): the protocol labels are inconsistent (the text says X-Sub for spatial, but the table headers say X-Set). [reader-inferred from a text vs table mismatch]
- **Relation to project:** The most recent NTU-120 skeleton-occlusion work, and it contains CTR-GCN occlusion baselines that we can compare with our CTR-GCN body-part dropout numbers. Check its tables for the exact CTR-GCN figures before citing them. Skeleton only.

**25. Shao, D., Shi, M., & Liu, L. (2026). FineTec: Fine-grained action recognition under temporal corruption via skeleton decomposition and sequence completion. *Proceedings of the AAAI Conference on Artificial Intelligence, 40*(11), 8842-8850. https://doi.org/10.1609/aaai.v40i11.37838**
- Verification: VERIFIED (Crossref + arXiv:2512.25067). Level III. Read: FT-HTML.
- **WHY / HOW:** Skeleton sequences with randomly dropped frames (25 / 50 / 75%, zero-padded). The method has three stages: context-aware sequence completion, spatial decomposition, and physics-based acceleration estimation; a GCN head classifies.
- **WHAT (NTU-120 X-Sub, 25 / 50 / 75% drop):**
  - FineTec: 81.9 / 81.7 / 81.3.
  - CTR-GCN: 81.4 / 80.9 / 79.3.
  - The skeletons appear to be 2D with 17 joints, so the numbers are not comparable with Kinect 25-joint results. The conversion is not explained.
- **Method weaknesses** (empirical; read FT-HTML): the skeleton source and joint format for NTU are unstated, which limits comparability (checked: task definition, datasets, experiments). [reader-inferred]
- **Relation to project:** Gives a third-party CTR-GCN temporal-corruption baseline on NTU-120 X-Sub. It shows CTR-GCN is fairly robust to random frame drops, which contrasts with SHARDeg's finding that contiguous blocks hurt more. Our RGB temporal mask is contiguous.

### Theme 3: Corruption-robustness benchmarks for video action recognition

**26. Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1903.12261**
- Verification: PLAUSIBLE→VERIFIED (arXiv comment "ICLR 2019 camera-ready"; well-established). Level III. Read: ABS.
- **WHY / HOW / WHAT:** ImageNet-C has 15 corruptions × 5 severities in four groups: noise, blur, weather, digital. It defines the mCE and relative mCE metrics normalised to a reference model. ImageNet-P covers perturbations.
- **Method weaknesses:** not assessed (read scope: abstract_only).
- **Relation to project:** The template for our "type × 5 severities" design and for reporting relative robustness. Our Gaussian blur comes from its blur family.

**27. Yi, C., Yang, S., Li, H., Tan, Y.-P., & Kot, A. C. (2021). Benchmarking the robustness of spatial-temporal models against corruptions. In *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/2a38a4a9316c49e5a833517c45d31070-Abstract-round2.html**
- Verification: VERIFIED (proceedings page + arXiv:2110.06513). Level III. Read: FT-HTML.
- **WHY:** Image-corruption benchmarks ignore temporal corruption.
- **HOW:**
  - Mini Kinetics-C and Mini SSV2-C: 12 corruptions × 5 severities.
  - Spatial: shot noise, rain, fog, contrast, brightness, saturate.
  - Temporal: motion blur, frame-rate conversion, H.265 ABR/CRF, bit error, packet loss.
  - Metrics: mPC and rPC.
  - Models: S3D, I3D, **3D ResNet-18**, 3D ResNet-50, SlowFast, X3D, TAM, TimeSformer.
- **WHAT:**
  - Transformers are more robust than CNNs.
  - Higher clean accuracy goes with higher robustness, especially to temporal corruptions.
  - **Gaussian-noise training hurt 3D ResNet-18** (lower clean accuracy and mPC; worse on 10 of 12 corruptions).
  - Training on a single corruption does not generalise.
- **Method weaknesses** (empirical; read FT-HTML): checked, found: operational definitions and metrics. Kinetics/SSV2 only, with no human-centric occlusion. [reader-inferred]
- **Relation to project:** The closest video benchmark to our RGB axis, and it shares our backbone family (3D ResNet-18). Its negative result on noise training is a direct caution for M4 corruption-aware training. Its corruption set has **no actor occlusion**, which makes our person-box occlusion a novel axis.

**28. Schiappa, M. C., Biyani, N., Kamtam, P., Vyas, S., Palangi, H., Vineet, V., & Rawat, Y. S. (2023). A large-scale robustness analysis of video action recognition models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 14698-14708). https://doi.org/10.1109/CVPR52729.2023.01412**
- Verification: VERIFIED (Crossref + CVF Open Access + arXiv:2207.01398). Note: the CVF title starts with "A Large-Scale ..."; the arXiv title is "Large-scale ...". Level III. Read: FT-HTML.
- **WHY:** Robustness of video models to real-world distribution shift.
- **HOW:**
  - 90 perturbations × 5 severities, in five groups:
    - noise: Gaussian, shot, impulse, speckle;
    - blur: zoom, motion, defocus;
    - digital: JPEG, MPEG1/2;
    - temporal: sampling rate, reversal, jumbling, box jumbling, freezing;
    - camera motion: rotation, translation.
  - Benchmarks: HMDB51-P, UCF101-P, Kinetics400-P, SSv2-P, plus the real-world UCF101-DS.
  - Models: **R3D**, I3D, SlowFast, X3D, MViT, TimeSformer.
  - Metrics: absolute and relative robustness.
- **WHAT:**
  - Transformers are more robust, and pretraining helps them more.
  - Models are largely robust to temporal perturbations except on SSv2.
  - All models struggle with noise and blur.
  - Perturbation training helps CNNs more than transformers.
- **Method weaknesses** (empirical; read FT-HTML): checked, found: operational definitions, external validity (4 datasets plus a real-shift set). No occlusion category. [reader-inferred]
- **Relation to project:** Evaluates R3D under blur and temporal corruptions, which gives an external point of comparison for our R3D-18 blur and temporal-mask results. No NTU, no skeleton, no fusion, and no actor-box occlusion.

---

## Source Verification Report

### Overall assessment
**Sources reviewed:** 37 assessed. **Included:** 28. **Verified (DOI or proceedings):** 24. **Plausible (arXiv, web or docs only):** 4 (SHARDeg, Babey, Guo, ImageNet-C via arXiv). **Rejected as fabricated:** 0. **Excluded for scope or quality:** 9 (listed above).

### Source quality matrix (condensed)

| Source | Level | Venue | Author | Method | Currency | COI | Overall |
|---|---|---|---|---|---|---|---|
| NTU-60 2016 / NTU-120 2020 | VI | pass | pass | pass | pass (seminal) | pass | A |
| CTR-GCN 2021 / R(2+1)D 2018 | III | pass | pass | not assessed | pass (seminal) | pass | A |
| Kinect TrackingState doc | VII | pass (vendor) | n/a | n/a | pass | pass | B (documentation) |
| VPN 2020, PoseC3D 2022, MMNet 2023, π-ViT 2024 | III | pass | pass | pass / warn (MMNet abstract only) | pass | pass | A |
| ActionMAE 2023, AMR 2024 | III | pass | pass | pass | pass | pass | A |
| Guo 2026 (NHSJS) | III | **warn** (student journal) | warn (single author, no affiliation verified) | warn (single seed) | pass | pass | C, cite as grey literature |
| Babey 2025 | III | warn (workshop/preprint) | pass | not assessed | pass | pass | C |
| Trust Gates 2018, RA-GCN 2019/2021 | III | pass | pass | pass | pass (seminal for protocol) | pass | A |
| PeGCN 2022, Liu C. 2024 SP | III | pass | pass | not assessed | pass | pass | B |
| SMART 2021, Tanaka 2024, Peng 2023, Chen 2025 | III | pass | pass | pass | pass | pass | A |
| SHARDeg 2025 | III | **warn** (preprint, unreviewed) | pass (Turing Inst. / ARL) | pass | pass | pass | B |
| Parts-Mamba 2026, FineTec 2026 | III | pass | pass | warn (protocol inconsistencies) | pass | pass | A- |
| ImageNet-C 2019, Yi 2021, Schiappa 2023 | III | pass | pass | pass | pass | pass | A |

### Flagged sources
- **MMNet:** author and year mismatch in the brief ("Bruce et al., TPAMI 2022"). Severity: Medium (citation accuracy). Recommendation: cite as Yu et al. (2023) and note the 2022 early access. The NTU-120 numbers are secondary (Papers with Code / Sensors 2024 table). Confirm them in the PDF before quoting.
- **SHARDeg:** preprint only. Severity: Low. Include with an "arXiv preprint" label.
- **Guo (2026), NHSJS:** student journal; peer review not confirmed. Severity: Medium. Include with a caveat, or cite only in a footnote.
- **PoseC3D numbers:** read from the ar5iv render of the arXiv version. NTU-120 95.3 / 96.4 is consistent with the widely cited CVPR figures. Its Table 7 NTU-60 value (96.2) differs from Table 9 (97.0) without explanation inside the paper.
- **Parts-Mamba / FineTec:** internal protocol and table inconsistencies (see entries). Severity: Low.

### Predatory journal alerts
None. All journals and conferences are IEEE, Springer, Elsevier, AAAI, NeurIPS, ICLR or CVF venues. NHSJS is not predatory but is a student venue (flagged above).

### Conflict-of-interest disclosures
No financial COI was identified. SHARDeg acknowledges funding from the US Army Research Laboratory (an ARL co-author); no conflict with the findings is apparent. Method papers carry the usual intellectual COI of benchmarking one's own method.

### Verification limitations
- Semantic Scholar was rate-limited (HTTP 429), so Tier 0 was skipped. Tier 1 (Crossref DOI/metadata) and Tier 2 (WebSearch / arXiv) were applied to **all** 28 sources rather than a 50% sample.
- The MMNet, Trust Gates, PeGCN, Liu C. 2024, CTR-GCN, R(2+1)D, RA-GCN ICIP, ImageNet-C and Babey entries were read at abstract/metadata level only. No method-level weaknesses are claimed for them.
- "Full text" means WebFetch extraction of HTML renders. Numbers were transcribed from those extractions and should be spot-checked against the PDFs before final citation.
- Absence claims are bounded by the searches listed above:
  - no RGB + skeleton corruption benchmark on NTU-120;
  - no use of the NTU trackingState as a natural-occlusion split;
  - no actor-box occlusion in the video corruption benchmarks.
  Recent (2026) and non-English work could have been missed.

### Search limitations
- WebSearch is US-indexed and gives no exact hit counts, so the PRISMA counts are approximate.
- Google Scholar and IEEE Xplore full-text search were not available. Paywalled pages (ScienceDirect, Springer) blocked abstract access for two candidates.
- Adjacent themes (uncertainty estimation, evidential or reliability-weighted fusion in general) were out of scope for this bibliography and are presumably covered by a sibling stage.

**Hand-off:** Phase 2 deliverable complete. Synthesis (Phase 3) should be routed to the synthesis stage and is not done here.
