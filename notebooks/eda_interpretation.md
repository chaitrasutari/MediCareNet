# MediCareNet EDA Interpretation Summary

## 1. Target Variable – `readmitted_30`

**Class Imbalance**  
- ~87% not readmitted, ~13% readmitted within 30 days.

**Implication for Modeling**  
- Avoid using plain accuracy.
- Instead use:
  - AUC-ROC
  - Precision/Recall
  - F1-score
  - Stratified k-fold cross-validation

---

## 2. Numeric Features – Distribution Analysis

From `numeric_distributions.png`:

| Feature              | Distribution Shape      | Insight                                 |
|----------------------|--------------------------|------------------------------------------|
| `num_lab_procedures` | Bell-shaped              | Useful, no need for binning              |
| `num_procedures`     | Skewed toward 0          | Many patients had no procedures          |
| `number_emergency`   | Highly skewed (0-heavy)  | Sparse signal; might not help alone      |
| `num_medications`    | Wide, slightly skewed    | Could have predictive power              |
| `time_in_hospital`   | Slight right skew        | Consider normalization                   |
| `number_inpatient`   | Mostly 0–1, some outliers| Potential for binning                    |

**Action:** Normalize or bin skewed features, especially for tree-based models and logistic regression.

---

## 3. Categorical Features – Top Breakdown

From `categorical_counts.png`:

- `race`: Dominated by Caucasian & African-American
- `age`: Normally distributed across bins – no transformation needed
- `diag_1`, `diag_2`, `diag_3`: Sparse, high-cardinality – already mapped to disease categories 
- `diabetes medications`: Mostly “No” or “Steady”

 **Action:** Simplify medication features into:
- “Used” vs “Not Used”
- Group rare medication levels into “Other” to reduce dimensionality

---

## 4. Correlation Heatmap

From `correlation_heatmap.png`:

**Top correlated with target (`readmitted_30`):**
- `number_inpatient` (0.17)
- `time_in_hospital` (0.06)
- `num_medications` (0.06)

**High inter-feature correlation:**
- `num_procedures` ↔ `num_medications`, `time_in_hospital`
- `number_inpatient`, `number_outpatient`, `number_emergency` are weakly correlated but complementary
**Action:**
- Feature engineering: Create composite feature like:
```python
df['total_visits'] = df['number_inpatient'] + df['number_outpatient'] + df['number_emergency']
```
- Remove or reduce multicollinear variables for linear models (e.g., Logistic Regression)


## 5. Target Leakage – `discharge_disposition_id`

**Result:**
- `discharge_disposition_id` values like **11 (Expired)**, **13/14 (Hospice)**, **19/20/21 (Non-return)** showed **zero readmissions**, indicating target leakage.

**Modeling Recommendation:**
- Drop records with leakage IDs:
```python
leak_ids = [11, 13, 14, 19, 20, 21]
df = df[~df['discharge_disposition_id'].isin(leak_ids)]
```
- Or group them into a 'Non-Return' class if needed.

---

## 6. KDE Distributions – Numeric Feature Comparison by `readmitted_30`

**Result:**
- `number_inpatient`, `num_medications`, and `time_in_hospital` showed clear differences in distribution between classes.

**Modeling Recommendation:**
- Retain all 5 examined features.
- Apply scaling if using linear models.
- Consider binning `number_inpatient` (e.g., `0`, `1–2`, `3+`).

---

## 7. Multicollinearity – VIF Check

**Result:**
- All numeric features had **VIF < 2**, indicating no multicollinearity issues.

**Modeling Recommendation:**
- Safe to use all numeric features in regression-based models.

---

## 8. Outlier Detection

**Result:**
- `num_medications` and `number_inpatient` showed strong outliers.
- `time_in_hospital` had mild outliers.

**Modeling Recommendation:**
- Cap `num_medications` at the 95th percentile.
- Bin `number_inpatient` into categories.
- No action needed for `time_in_hospital`.

---

## 9. Rare Category Analysis

**Result:**
- Rare values (<1%) found in columns like `repaglinide`, `tolazamide`, `nateglinide`, etc.

**Modeling Recommendation:**
- Group rare categories as `'Other'` prior to encoding.
- Use provided map to apply grouping.

---

## 10. Feature-Target Relationships – Categorical Stacked Bars

**Result:**
- `age` showed a strong upward trend in readmission.
- `race` and `gender` had minimal differences.

**Modeling Recommendation:**
- Retain `age`, possibly bin.
- Retain `race` for fairness tracking.
- Gender is optional.

---

## 11. Interaction Feature – `age × number_inpatient`

**Result:**
- Younger patients with high inpatient counts had highest readmission.
- Interaction pattern visible across all age groups.

**Modeling Recommendation:**
- Create interaction feature if using non-tree models:
```python
df['age_inpatient_interaction'] = df['age'].astype(str) + '_' + pd.cut(df['number_inpatient'], bins=[-1, 0, 2, 100], labels=["0", "1-2", "3+"]).astype(str)
```
- Tree models will capture interaction implicitly.

---

## 12. Interaction Feature – `diabetesMed × insulin`

**Result:**
- Highest readmission rates seen in patients with `diabetesMed = Yes` and `insulin` values `Down` or `Up`.

**Modeling Recommendation:**
- Create an interaction feature:
```python
df['diabetes_insulin_combo'] = df['diabetesMed'] + '_' + df['insulin']
```

---

## 13. Bias & Fairness Analysis

**Result:**
- **Race**: Slight differences (e.g., AfricanAmerican slightly higher)
- **Gender**: Nearly identical
- **Age**: Clear peak at [20–30) (~14.4%), low for <20

**Modeling Recommendation:**
- Log group-wise performance (AUC, F1) for `race` and `age`
- Track fairness metrics in production (e.g., demographic parity)
- Consider SHAP for post-hoc audit
