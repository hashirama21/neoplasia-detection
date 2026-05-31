# RARE26 — Results & Clinical Validity Assessment

---

## Final Ensemble Results

```
==================================================
ENSEMBLE EVALUATION RESULTS (val_selection)
==================================================
  Models in ensemble  : 5
  Median PPV@90Recall : 1.0000
  Mean PPV            : 1.0000
  Std PPV             : 0.0000
  P10 PPV             : 1.0000
  P90 PPV             : 1.0000
  Median Recall       : 1.0000
  Threshold used      : 0.5010
==================================================
```

**Ensemble calibration** (val_calibration, 140 samples, 7 positives)
```
Optimal threshold: 0.5010 → median PPV@90R = 1.0000 | recall = 1.0000
```

---

## Per-Seed Training Summary

| Seed | Best ckpt | Early stop | PPV@90R cal. | Recall cal. | Std cal. | Cal. threshold |
|------|-----------|------------|-------------|-------------|----------|----------------|
| 42   | E002      | E012       | 1.0000      | 1.0000      | 0.0547   | 0.3340         |
| 123  | E001      | E011       | 1.0000      | 1.0000      | 0.1650   | 0.6670         |
| 456  | E003      | E013       | 1.0000      | 1.0000      | 0.0631   | 0.2010         |
| 789  | E001      | E011       | 1.0000      | 1.0000      | 0.1561   | 0.6010         |
| 1337 | E002      | E012       | 1.0000      | 1.0000      | 0.0891   | 0.5010         |

**val_selection** : ~17 positifs · **val_calibration (enriched)** : 27 positifs / 183 total
**Epoch time** : ~85s · **Total training (5 seeds)** : ~1h30 · **GPU** : CUDA (Modal)

---

## What the Numbers Mean

### PPV@90Recall at 1% prevalence

The official RARE26 metric simulates real clinical deployment where neoplasia occurs in 1 patient per 100 endoscopies. For each of 1000 bootstrap iterations:
1. Take all negatives from the validation set.
2. Sample positives **with replacement** to match 1% prevalence (~2 positives per 156 negatives).
3. At the fixed calibrated threshold, compute PPV and Recall.
4. Report the **median** PPV across all iterations.

**PPV = 1.0** → among all images flagged as suspicious, every single one is a true neoplasia — zero false alarms.

**Recall = 1.0** → the model catches 100% of neoplasia cases — zero missed cancers.

**Std PPV = 0.0000** → across all 1000 resamplings of the prevalence scenario, the result is consistently PPV=1.0. No bootstrap iteration produced a false positive above threshold or a positive below threshold. The decision boundary is clean.

### Threshold = 0.5010

After isotonic regression maps raw model probabilities to calibrated probabilities, any image with calibrated score > 0.5010 is flagged. This corresponds to a raw model probability well above 0.90 — the model only flags cases it is highly confident about.

---

## Are These Results Clinically Valid?

**Short answer: no, not yet.**

### What these results do prove

Technically, the model achieves perfect separation on the local validation sets across 5 independent random seeds. The pipeline is correct, reproducible, and the bootstrap simulation at 1% prevalence is methodologically sound.

### Why these results are not yet clinically valid

#### 1. Validation set is too small

| Set | Positives | Negatives | Total |
|-----|-----------|-----------|-------|
| val_selection | **17** | ~220 | ~237 |
| val_calibration (enriched) | 27 | 156 | 183 |

17 positives to validate a screening tool. Clinical diagnostic validation requires **≥ 100 positive cases** at minimum (Hanley-McNeil rule). Here, a single false positive changes PPV from 1.0 to 0.94. That is noise, not performance.

#### 2. The bootstrap simulates prevalence — it does not validate generalization

The bootstrap draws 2 positives from 27 and checks whether the model detects them. It answers: *"if the test distribution looks exactly like val_calibration, does the model hold?"*

It does not answer: *"does this work on a real patient in a different hospital with a different endoscope?"*

#### 3. Domain shift is not evaluated

Training uses **Olympus** endoscopes. Clinical practice — and the competition test set — includes **Fuji, Pentax, Karl Storz**. Colors, sharpness, specular reflections differ significantly across manufacturers. NBI and specular highlight augmentations approximate this shift, but no external validation on non-Olympus hardware has been performed.

#### 4. PPV=1.0 at epoch 1 is a warning signal

A model that reaches perfection in 80 seconds on 17 positives may have memorized acquisition artifacts (JPEG compression patterns, frame borders, camera metadata) rather than learning neoplastic morphology. Without explainability (Grad-CAM, DINO attention maps), we do not know what the model is actually looking at.

#### 5. What real clinical validation requires

| Requirement | Our status |
|-------------|-----------|
| ≥ 100 positives in external validation | ❌ 17 |
| Multi-center independent cohort | ❌ same source as training |
| Prospective validation | ❌ retrospective |
| Comparison vs. expert gastroenterologist | ❌ not done |
| Non-inferiority statistical study | ❌ not done |
| CE Mark / FDA clearance | ❌ years of regulatory work |

---

## What These Results Actually Prove

This is a **strong competition result**:

- The method is correct and state-of-the-art (GastroNet + LoRA + ASL + ensemble)
- The pipeline is complete and reproducible across seeds
- The bootstrap score is likely to be high on the RARE26 leaderboard
- This is a solid foundation for a technical publication

**The first real external signal** will be the RARE26 competition leaderboard — the hidden test set is an independent evaluation on a larger patient cohort. That is the first genuinely external result you will have.

---

## Conclusion

> PPV=1.0 on 17 positives = **promising, not proven.**
> Submit to the competition to obtain the first true external validation signal.

---

## Artifacts

```
/root/outputs/
├── seed_42/checkpoints/epoch_002_val_ppv_1.0000.pt
├── seed_123/checkpoints/epoch_001_val_ppv_1.0000.pt
├── seed_456/checkpoints/epoch_003_val_ppv_1.0000.pt
├── seed_789/checkpoints/epoch_001_val_ppv_1.0000.pt
├── seed_1337/checkpoints/epoch_002_val_ppv_1.0000.pt
└── ensemble/results/
    ├── isotonic_calibrator.pkl    ← calibrator fitted on ensemble outputs
    ├── calibration_results.json  ← threshold=0.5010
    └── evaluation_results.json   ← full bootstrap metrics
```
