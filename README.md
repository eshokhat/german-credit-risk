# Fair Credit Assessment: Bias Mitigation in Machine Learning

## Summary

This project investigates demographic bias in credit scoring systems using the
[Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
from the UCI Machine Learning Repository. It explores how a Random Forest
classifier trained on historical bank decisions inherits and amplifies
demographic disparities — specifically for young applicants (Age ≤ 25) — and
evaluates three fairness interventions applied at different stages of the
machine learning lifecycle.

---

## Project Motivation

Financial institutions increasingly rely on automated models to determine
creditworthiness. However, models trained on historical data often inherit or
amplify the biases embedded in those human-driven decisions. This project aims to:

1. Quantify the demographic parity gap present in historical bank data.
2. Demonstrate how a baseline model propagates and amplifies this gap.
3. Evaluate the effectiveness of pre-training, in-training, and post-training
   interventions in closing the fairness gap.

---

## Methodology

### 1. Data Preparation and Target Variable

The project uses 1,000 applicant records from the German Credit dataset. The
target variable is defined directly from the bank's historical credit ratings:

- **y = 1**: The bank rated the applicant as **Bad** (high credit risk).
- **y = 0**: The bank rated the applicant as **Good** (creditworthy).

No simulation of default outcomes is introduced. The model learns to reproduce
the bank's classification logic, and fairness is measured as the degree to
which those reproduced decisions are applied equally across age groups.

Exploratory Data Analysis identified approval rate gaps across three demographic
attributes. Chi-squared tests (α = 0.05) were applied to each contingency table
to verify statistical significance before drawing any conclusions about focus:

| Attribute | Gap (pp) | p-value | Significant? |
|---|---|---|---|
| Sex (Male vs Female) | 7.5 | 0.0170 | Yes |
| Age (Young ≤25 vs Other) | 14.9 | 0.0001 | Yes |
| Marital Status (max spread) | 13.4 | 0.0222 | Yes |

All three gaps are statistically significant. The project nonetheless focuses on
**age as the sole protected attribute** for the following reasons:

1. **Strength of evidence**: Age's p-value (0.0001) is roughly 170–430× smaller
   than those for sex and marital status. The signal is categorically stronger,
   not merely incrementally so.
2. **Effect size**: Age also has the largest absolute gap (14.9 pp), so both
   statistical and practical significance point in the same direction.
3. **Structural collinearity of sex and marital status**: Both variables are
   decoded from the same raw field (`Attribute9`). The female category maps
   exclusively onto two marital-status codes, meaning the marital-status gap is
   not independent of the sex gap — they are partial views of the same
   underlying attribute. Treating them as three separate significant findings
   would be misleading.

This is a principled narrowing of scope, not an omission. The p-values above
are reported in full so that readers can assess the evidence for all three
attributes.

### 2. Baseline Model

A Random Forest classifier is trained to replicate the bank's decision logic
and evaluated on two dimensions:

- **Predictive performance**: ROC-AUC and Accuracy measure how closely the
  model reproduces bank decisions.
- **Fairness audit**: The Demographic Parity gap is defined as the absolute
  difference in Bad-label rates between young applicants (Age ≤ 25) and the
  reference group (Age > 25).

The baseline model (ROC-AUC = 0.793, Accuracy = 0.665) successfully learned the
bank's decision boundaries but amplified the existing age disparity:

| | DP gap (age) |
|---|---|
| Bank (original) | 0.1981 |
| Baseline RF | 0.2080 (+0.010) |

At the decision threshold of 0.4, the baseline model's confusion matrix on the
200-applicant test set is: 80 true negatives, 60 false positives, 7 false
negatives, and 53 true positives (Bad-label recall = 0.88).

### 3. Fairness Interventions

Three interventions are tested, one at each stage of the ML lifecycle. All
use `is_young` (Age ≤ 25) as the sensitive attribute.

- **Pre-training — Reweighing**: Training sample weights are adjusted using
  the Kamiran & Calders (2012) formula so that the joint distribution of
  `(is_young, label)` is uniform before training begins.
- **In-training — Cost-sensitive weights**: The Random Forest is trained with
  `class_weight = {0: 1, 1: 5}`, encoding the UCI cost asymmetry directly
  into the splitting criterion.
- **Post-training — Group thresholds**: Separate classification thresholds are
  applied after training — a higher threshold for young applicants
  (`THRESHOLD_YOUNG = 0.40`) and a standard threshold for others
  (`THRESHOLD_OTHER = 0.30`). No retraining is required.

| Model | ROC-AUC | Accuracy | DP gap (age) |
|---|---|---|---|
| Baseline RF | 0.7933 | 0.665 | 0.2080 |
| Pre-Training (Reweighing) | 0.7908 | 0.640 | 0.1142 |
| In-Training (Cost Weights 1:5) | 0.7930 | 0.490 | 0.1783 |
| Post-Training (Group Thresholds) | 0.7933 | 0.525 | 0.0157 |

### 4. Feature Importance

Mean Decrease in Impurity (MDI) is aggregated across one-hot encoded columns
to show the contribution of each original attribute. This reveals whether the
model relies on financial behavioural signals or on protected demographic
features, providing transparency into the mechanism of learned bias.

| Feature | Aggregated MDI share |
|---|---|
| Checking Account Status | 0.2066 |
| Loan Duration (months) | 0.0896 |
| Loan Amount | 0.0888 |
| Credit History | 0.0669 |
| Savings Account | 0.0653 |
| Property | 0.0589 |
| **Age (years)** | **0.0572** |
| Loan Purpose | 0.0563 |
| Employment Duration | 0.0557 |
| Other Installment Plans | 0.0360 |
| Installment Rate (% income) | 0.0274 |
| Housing | 0.0271 |
| Residence Duration (years) | 0.0251 |
| Job Type | 0.0247 |
| Telephone Registered | 0.0205 |

Age ranks 7th in aggregated MDI share (0.057), ahead of several features with
clear financial content (Loan Purpose, Employment Duration). This confirms that
the model uses age directly as a predictive signal — the structural cause of
the amplified DP gap.

---

## Key Findings

- **Bias amplification**: The baseline model increases the age-based DP gap
  from the bank's 0.198 to 0.208. A model trained on historical decisions does
  not reproduce them neutrally — it applies the embedded pattern more
  consistently than any individual human decision-maker.
- **All interventions reduce the gap below bank level**: Each of the three
  methods brings the DP gap below the bank's original 0.198, demonstrating
  that fairness improvements are achievable without discarding the model.
- **Best result**: Post-training group thresholds achieve the lowest DP gap
  (0.016) — effectively closing the age gap — while maintaining the same
  ROC-AUC as the baseline (0.793), at a larger accuracy cost than reweighing
  (0.525 vs 0.640). The threshold values are exposed as variables in the
  notebook and can be adjusted to explore the fairness-accuracy tradeoff.
- **Age as a model feature**: Age accounts for 5.7% of total feature
  importance — confirming it is not only a correlate of financial risk but a
  direct input to the model's decisions.

---

## Conclusion

A Random Forest trained on historical bank credit decisions inherits and
amplifies the age-based disparity present in the training data. However, all
three lifecycle interventions successfully reduce the DP gap below the bank's
own level. The best result (DP gap = 0.016, achieved by post-training threshold
adjustment) represents a 92% reduction relative to the bank's original gap of
0.198 — without any loss in discriminative power (ROC-AUC = 0.793).

This confirms that fairness improvements are achievable within the existing
modelling framework. Deploying a credit scoring model without fairness
evaluation is not a neutral act: it encodes historical bias into future
decisions at scale.

---

## References

- Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5NC77
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for
  classification without discrimination. *Knowledge and Information Systems,
  33*(1), 1–33.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in
  supervised learning. *Advances in Neural Information Processing Systems, 29*.
