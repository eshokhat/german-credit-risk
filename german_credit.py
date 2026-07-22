# -*- coding: utf-8 -*-
import warnings
import matplotlib
matplotlib.use('Agg')  # Настройка бэкенда для работы без GUI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# Seed воспроизводимости
RANDOM_STATE = 42

# ── Цветовая палитра ──────────────────────────────────────────────────────────
C_DENY    = '#EF4323'
C_APPROVE = '#5804FA'
C_BEST    = '#22E563'
C_INK     = '#131313'
C_BG      = '#F9F9F9'

# Глобальные настройки графиков
plt.rcParams.update({
    'figure.dpi'          : 120,
    'figure.facecolor'    : C_BG,
    'axes.facecolor'      : C_BG,
    'axes.edgecolor'      : C_INK,
    'axes.labelcolor'     : C_INK,
    'xtick.color'         : C_INK,
    'ytick.color'         : C_INK,
    'text.color'          : C_INK,
    'axes.grid'           : False,
    'font.family'         : 'sans-serif',
})

print('All imports OK')

# ── 1. Download Data ──────────────────────────────────────────────────────────
dataset = fetch_ucirepo(id=144)

X_raw = dataset.data.features.copy()   # 1000 × 20
y_raw = dataset.data.targets.copy()    # 1000 × 1  (column name: 'class')

y_raw = y_raw['class'].rename('class')

df = X_raw.copy()
df['class'] = y_raw.values

print(f'Dataset shape: {df.shape}')
print(f"Class distribution (1=Good, 2=Bad):\n{df['class'].value_counts()}")
print(df.head())
df.info()

# ── 2. Descriptive Analysis (pre-intervention) ────────────────────────────────
A9_SEX = {
    'A91': 'male',
    'A92': 'female',
    'A93': 'male',
    'A94': 'male',
    'A95': 'female',
}
A9_MARITAL = {
    'A91': 'divorced/separated',
    'A92': 'divorced/separated/married',
    'A93': 'single',
    'A94': 'married/widowed',
    'A95': 'single',
}
A9_LABEL = {
    'A91': 'M: div/sep',
    'A92': 'F: div/sep/mar',
    'A93': 'M: single',
    'A94': 'M: mar/wid',
    'A95': 'F: single',
}

df['_sex']      = df['Attribute9'].map(A9_SEX)
df['_marital']  = df['Attribute9'].map(A9_MARITAL)
df['_a9_label'] = df['Attribute9'].map(A9_LABEL)
df['_age']      = df['Attribute13'].astype(int)
df['_is_young'] = (df['_age'] <= 25).map({True: 'Age ≤ 25', False: 'Age > 25'})
df['_approved'] = (df['class'] == 1).map({True: 'Approved', False: 'Denied'})

print('Temporary EDA columns added.')

ATTR_LABEL = {
    'Attribute1'  : 'Checking Account Status',
    'Attribute2'  : 'Loan Duration (months)',
    'Attribute3'  : 'Credit History',
    'Attribute4'  : 'Loan Purpose',
    'Attribute5'  : 'Loan Amount',
    'Attribute6'  : 'Savings Account',
    'Attribute7'  : 'Employment Duration',
    'Attribute8'  : 'Installment Rate (% income)',
    'Attribute9'  : 'Personal Status & Sex',
    'Attribute10' : 'Other Debtors / Guarantors',
    'Attribute11' : 'Residence Duration (years)',
    'Attribute12' : 'Property',
    'Attribute13' : 'Age (years)',
    'Attribute14' : 'Other Installment Plans',
    'Attribute15' : 'Housing',
    'Attribute16' : 'Number of Existing Credits',
    'Attribute17' : 'Job Type',
    'Attribute18' : 'Number of Dependents',
    'Attribute19' : 'Telephone Registered',
    'Attribute20' : 'Foreign Worker',
    'is_young'    : 'Young Applicant (Age <= 25)',
    'sex_male'    : 'Sex: Male',
    'sex_female'  : 'Sex: Female',
    'marital_status_single'                    : 'Marital: Single',
    'marital_status_married/widowed'           : 'Marital: Married/Widowed',
    'marital_status_divorced/separated'        : 'Marital: Divorced/Separated',
    'marital_status_divorced/separated/married': 'Marital: Div/Sep/Married',
}

def readable(col):
    for key, label in ATTR_LABEL.items():
        if col == key or col.startswith(key + '_'):
            return label
    return col

# Plot Gender
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=C_BG)

sex_counts = df.groupby(['_sex', '_approved']).size().unstack(fill_value=0)
sex_counts.plot(kind='bar', ax=axes[0], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[0].set_title('Approval / Denial by Gender (counts)', fontweight='bold', color=C_INK)
axes[0].set_xlabel('Gender', color=C_INK)
axes[0].set_ylabel('Number of applicants', color=C_INK)
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='')
axes[0].grid(False)

sex_pct = sex_counts.div(sex_counts.sum(axis=1), axis=0) * 100
sex_pct.plot(kind='bar', stacked=True, ax=axes[1], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_title('Approval / Denial by Gender (%)', fontweight='bold', color=C_INK)
axes[1].set_xlabel('Gender', color=C_INK)
axes[1].set_ylabel('Share of applicants', color=C_INK)
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='', loc='lower right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('plots/plot_gender.png', bbox_inches='tight')
plt.close()

print(sex_pct.round(1).to_string())

# Plot Age Group
bins   = [0, 25, 35, 45, 55, 120]
labels = ['<=25', '26-35', '36-45', '46-55', '55+']
df['_age_group'] = pd.cut(df['_age'], bins=bins, labels=labels, right=True)

age_counts = df.groupby(['_age_group', '_approved'], observed=True).size().unstack(fill_value=0)
age_pct    = age_counts.div(age_counts.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=C_BG)

age_counts.plot(kind='bar', ax=axes[0], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[0].set_title('Approval / Denial by Age Group (counts)', fontweight='bold', color=C_INK)
axes[0].set_xlabel('Age Group', color=C_INK)
axes[0].set_ylabel('Number of applicants', color=C_INK)
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='')
axes[0].grid(False)

age_pct.plot(kind='bar', stacked=True, ax=axes[1], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_title('Approval / Denial by Age Group (%)', fontweight='bold', color=C_INK)
axes[1].set_xlabel('Age Group', color=C_INK)
axes[1].set_ylabel('Share of applicants', color=C_INK)
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='', loc='lower right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('plots/plot_age_group.png', bbox_inches='tight')
plt.close()

# Plot Marital Status
mar_counts = df.groupby(['_marital', '_approved']).size().unstack(fill_value=0)
mar_pct    = mar_counts.div(mar_counts.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor=C_BG)

mar_counts.plot(kind='bar', ax=axes[0], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[0].set_title('Approval / Denial by Marital Status (counts)', fontweight='bold', color=C_INK)
axes[0].set_xlabel('')
axes[0].set_ylabel('Number of applicants', color=C_INK)
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend(title='')
axes[0].grid(False)

mar_pct.plot(kind='bar', stacked=True, ax=axes[1], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_title('Approval / Denial by Marital Status (%)', fontweight='bold', color=C_INK)
axes[1].set_xlabel('')
axes[1].set_ylabel('Share of applicants', color=C_INK)
axes[1].tick_params(axis='x', rotation=30)
axes[1].legend(title='', loc='lower right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('plots/plot_marital.png', bbox_inches='tight')
plt.close()

def chi2_pvalue(df_src, group_col, outcome_col, approved_label='Approved'):
    ct = pd.crosstab(df_src[group_col], df_src[outcome_col] == approved_label)
    _, p, _, _ = stats.chi2_contingency(ct, correction=False)
    return p

gap_rows = []

sex_approved = df.groupby('_sex')['_approved'].apply(lambda s: (s == 'Approved').mean() * 100)
gap_rows.append({
    'Attribute'     : 'Sex (Male vs Female)',
    'Group A'       : 'Male',
    'Approval A (%)': round(sex_approved.get('male',   0), 1),
    'Group B'       : 'Female',
    'Approval B (%)': round(sex_approved.get('female', 0), 1),
    'Gap (pp)'      : round(abs(sex_approved.get('male', 0) - sex_approved.get('female', 0)), 1),
    'p-value'       : chi2_pvalue(df, '_sex', '_approved'),
})

young_approved = df.groupby('_is_young')['_approved'].apply(lambda s: (s == 'Approved').mean() * 100)
gap_rows.append({
    'Attribute'     : 'Age (Young ≤25 vs Other)',
    'Group A'       : 'Age > 25',
    'Approval A (%)': round(young_approved.get('Age > 25', 0), 1),
    'Group B'       : 'Age ≤ 25',
    'Approval B (%)': round(young_approved.get('Age ≤ 25', 0), 1),
    'Gap (pp)'      : round(abs(young_approved.get('Age > 25', 0) - young_approved.get('Age ≤ 25', 0)), 1),
    'p-value'       : chi2_pvalue(df, '_is_young', '_approved'),
})

mar_approved = df.groupby('_marital')['_approved'].apply(lambda s: (s == 'Approved').mean() * 100)
gap_rows.append({
    'Attribute'     : 'Marital Status (max spread)',
    'Group A'       : mar_approved.idxmax(),
    'Approval A (%)': round(mar_approved.max(), 1),
    'Group B'       : mar_approved.idxmin(),
    'Approval B (%)': round(mar_approved.min(), 1),
    'Gap (pp)'      : round(mar_approved.max() - mar_approved.min(), 1),
    'p-value'       : chi2_pvalue(df, '_marital', '_approved'),
})

gap_df = pd.DataFrame(gap_rows).set_index('Attribute')

print("=" * 78)
print("  EDA SUMMARY — Approval Rate Gaps by Demographic Attribute")
print("=" * 78)
print(gap_df.to_string())
print()
print("Interpretation (significance threshold α = 0.05):")
for attr, row in gap_df.iterrows():
    sig = "  *" if row['p-value'] < 0.05 else "   (not significant)"
    print(f"  {attr:<38} gap = {row['Gap (pp)']:>5.1f} pp   p = {row['p-value']:.4f}{sig}")

# Plot Young vs Others
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=C_BG)

young_counts = df.groupby(['_is_young', '_approved']).size().unstack(fill_value=0)
young_pct    = young_counts.div(young_counts.sum(axis=1), axis=0) * 100

young_counts.plot(kind='bar', ax=axes[0], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[0].set_title('Approval / Denial: Young vs Others (counts)', fontweight='bold', color=C_INK)
axes[0].set_xlabel('')
axes[0].set_ylabel('Number of applicants', color=C_INK)
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(title='')
axes[0].grid(False)

young_pct.plot(kind='bar', stacked=True, ax=axes[1], color=[C_APPROVE, C_DENY], edgecolor='white')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_title('Approval / Denial: Young vs Others (%)', fontweight='bold', color=C_INK)
axes[1].set_xlabel('')
axes[1].set_ylabel('Share of applicants', color=C_INK)
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(title='', loc='lower right')
axes[1].grid(False)

plt.tight_layout()
plt.savefig('plots/plot_young.png', bbox_inches='tight')
plt.close()

print(young_pct.round(1).to_string())
approval_gap = young_pct.loc['Age > 25', 'Approved'] - young_pct.loc['Age ≤ 25', 'Approved']
print(f"\nApproval gap: {young_pct.loc['Age > 25','Approved']:.1f}% vs "
      f"{young_pct.loc['Age ≤ 25','Approved']:.1f}% = {approval_gap:.1f} pp")

# ── 3. Feature Engineering ────────────────────────────────────────────────────
feat = X_raw.copy()
feat['sex'] = feat['Attribute9'].map(A9_SEX)
feat['marital_status'] = feat['Attribute9'].map(A9_MARITAL)
feat.drop(columns=['Attribute9'], inplace=True)

feat['is_young'] = (feat['Attribute13'].astype(int) <= 25).astype(int)

cat_cols = feat.select_dtypes(include=['object', 'category']).columns.tolist()
feat_encoded = pd.get_dummies(feat, columns=cat_cols, drop_first=False, dtype=int)

print(f'\nShape after encoding: {feat_encoded.shape}')
print(feat_encoded.head())

# ── 4. Target Variable ────────────────────────────────────────────────────────
gt_df = feat_encoded.copy()
gt_df['class'] = y_raw.values
gt_df['y']     = (gt_df['class'] == 2).astype(int)   # 1 = bank said Bad

print(f"Total applicants      : {len(gt_df)}")
print(f"Bank-rated Bad  (y=1) : {gt_df['y'].sum()}  ({gt_df['y'].mean():.1%})")
print(f"Bank-rated Good (y=0) : {(gt_df['y'] == 0).sum()}  ({(gt_df['y'] == 0).mean():.1%})")

# ── 5. Train / Test Split ─────────────────────────────────────────────────────
feature_cols = [c for c in feat_encoded.columns]

X_all  = gt_df[feature_cols]
y_gt   = gt_df['y']
y_bank = gt_df['class']

X_train, X_test, y_train, y_gt_test, y_bank_train, y_bank_test = train_test_split(
    X_all, y_gt, y_bank,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_gt,
)

assert X_train.isnull().sum().sum() == 0, 'NaNs in X_train!'
assert X_test.isnull().sum().sum()  == 0, 'NaNs in X_test!'
assert y_train.isnull().sum()       == 0, 'NaNs in y_train!'
assert y_gt_test.isnull().sum()     == 0, 'NaNs in y_gt_test!'

non_numeric = X_train.select_dtypes(exclude=['number']).columns.tolist()
assert len(non_numeric) == 0, f'Non-numeric columns remain: {non_numeric}'

print(' All assertions passed.')

# ── STAGE 2: Baseline Model ───────────────────────────────────────────────────
def dp_gap(y_pred, sensitive_col):
    groups = sorted(sensitive_col.unique())
    rows   = []
    for g in groups:
        mask      = (sensitive_col == g)
        flag_rate = pd.Series(y_pred, index=sensitive_col.index)[mask].mean()
        rows.append({'group': g, 'n': int(mask.sum()), 'flag_rate': flag_rate})
    df_g = pd.DataFrame(rows).set_index('group')
    df_g['DP_gap'] = abs(df_g['flag_rate'].iloc[0] - df_g['flag_rate'].iloc[1])
    return df_g

rf_baseline = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_leaf=7,
    max_features='log2',
    criterion='entropy',
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

rf_baseline.fit(X_train, y_train)

THRESHOLD = 0.4
y_prob_baseline = rf_baseline.predict_proba(X_test)[:, 1]
y_pred_baseline = (y_prob_baseline >= THRESHOLD).astype(int)

auc_train    = roc_auc_score(y_train, rf_baseline.predict_proba(X_train)[:, 1])
auc_test     = roc_auc_score(y_gt_test, y_prob_baseline)
acc_baseline = accuracy_score(y_gt_test, y_pred_baseline)
cm_baseline  = confusion_matrix(y_gt_test, y_pred_baseline)

tn, fp, fn, tp = cm_baseline.ravel()

print("=" * 52)
print("  BASELINE MODEL — PERFORMANCE METRICS")
print("=" * 52)
print(f"  ROC-AUC  (train) : {auc_train:.4f}")
print(f"  ROC-AUC  (test)  : {auc_test:.4f}")
print(f"  Accuracy (test)  : {acc_baseline:.4f}")
print("-" * 52)
print(classification_report(y_gt_test, y_pred_baseline, target_names=['Good (0)', 'Bad (1)']))

bank_pred_binary = (y_bank_test == 2).astype(int).values
auc_bank  = roc_auc_score(y_gt_test, bank_pred_binary)
acc_bank  = accuracy_score(y_gt_test, bank_pred_binary)

young_test  = X_test['is_young']
dp_age_bank = dp_gap(bank_pred_binary, young_test)
dp_age_base = dp_gap(y_pred_baseline,  young_test)

gap_bank = dp_age_bank['DP_gap'].iloc[0]
gap_base = dp_age_base['DP_gap'].iloc[0]

# Stage 2 Plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=C_BG)

ax = axes[0]
outcomes = ['True Neg\n(correct Good)', 'False Pos\n(over-flagged)',
            'False Neg\n(missed Bad)',  'True Pos\n(correct Bad)']
bars = ax.bar(outcomes, [tn, fp, fn, tp], color=[C_APPROVE, C_DENY, C_DENY, C_BEST], edgecolor='white', width=0.55)
ax.bar_label(bars, padding=3, fontsize=10, fontweight='bold')
ax.set_title(f'Prediction Outcomes (threshold = {THRESHOLD})', fontweight='bold', color=C_INK)
ax.set_ylabel('Number of applicants', color=C_INK)
ax.tick_params(axis='x', labelsize=8)
ax.grid(False)

ax = axes[1]
bars2 = ax.bar(['Bank decisions', 'Baseline RF'], [gap_bank, gap_base], color=[C_DENY, C_APPROVE], edgecolor='white', width=0.40)
ax.bar_label(bars2, fmt='%.3f', padding=4, fontsize=11, fontweight='bold')
ax.set_title('Demographic Parity Gap — Age\n(lower = more equal treatment)', fontweight='bold', color=C_INK)
ax.set_ylabel('Absolute gap in Bad-label rates', color=C_INK)
ax.axhline(0.1, color=C_DENY, linestyle=':', lw=1.2, label='0.1 reference')
ax.set_ylim(0, max(gap_bank, gap_base) * 1.3)
ax.legend(fontsize=9)
ax.grid(False)

plt.suptitle('Stage 2 — Baseline Model: Performance and Fairness (Age)', fontsize=12, fontweight='bold', y=1.01, color=C_INK)
plt.tight_layout()
plt.savefig('plots/stage2_baseline_audit.png', bbox_inches='tight')
plt.close()

# ── STAGE 3: Fairness Interventions ───────────────────────────────────────────
def collect_metrics(name, y_true, y_prob, y_pred, s_young):
    auc  = roc_auc_score(y_true, y_prob)
    acc  = accuracy_score(y_true, y_pred)
    dp_y = dp_gap(y_pred, s_young)['DP_gap'].iloc[0]
    return {
        'Model'        : name,
        'ROC-AUC'      : round(auc,  4),
        'Accuracy'     : round(acc,  4),
        'DP gap (age)' : round(dp_y, 4),
    }

results_log = [
    collect_metrics('Bank (original)', y_gt_test, bank_pred_binary.astype(float), bank_pred_binary, young_test),
    collect_metrics('Baseline RF', y_gt_test, y_prob_baseline, y_pred_baseline, young_test)
]

# 3.2 Pre-Training: Reweighing
def compute_reweighing_weights(y: pd.Series, sensitive: pd.Series) -> np.ndarray:
    df_w = pd.DataFrame({'y': y.values, 's': sensitive.values})
    n    = len(df_w)
    p_y  = df_w['y'].value_counts(normalize=True)
    p_s  = df_w['s'].value_counts(normalize=True)
    p_sy = df_w.groupby(['s', 'y']).size() / n
    return df_w.apply(lambda row: (p_s[row['s']] * p_y[row['y']]) / p_sy[(row['s'], row['y'])], axis=1).values

young_train = X_train['is_young']
sample_weights_rw = compute_reweighing_weights(y_train, young_train)

rf_reweigh = RandomForestClassifier(
    n_estimators=250, max_depth=10, min_samples_leaf=7,
    max_features='log2', criterion='entropy',
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf_reweigh.fit(X_train, y_train, sample_weight=sample_weights_rw)

y_prob_rw = rf_reweigh.predict_proba(X_test)[:, 1]
y_pred_rw = (y_prob_rw >= THRESHOLD).astype(int)
results_log.append(collect_metrics('Pre-Training (Reweighing)', y_gt_test, y_prob_rw, y_pred_rw, young_test))

# 3.3 In-Training: Cost-Sensitive
rf_cost = RandomForestClassifier(
    n_estimators=250, max_depth=10, min_samples_leaf=7,
    max_features='log2', criterion='entropy',
    class_weight={0: 1, 1: 5},
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf_cost.fit(X_train, y_train)

y_prob_cost = rf_cost.predict_proba(X_test)[:, 1]
y_pred_cost = (y_prob_cost >= THRESHOLD).astype(int)
results_log.append(collect_metrics('In-Training (Cost Weights 1:5)', y_gt_test, y_prob_cost, y_pred_cost, young_test))

# 3.4 Post-Training: Thresholds
THRESHOLD_YOUNG = 0.4
THRESHOLD_OTHER = 0.3

is_young_test = (young_test == 1).values
y_pred_thresh = np.where(
    is_young_test,
    (y_prob_baseline >= THRESHOLD_YOUNG).astype(int),
    (y_prob_baseline >= THRESHOLD_OTHER).astype(int),
)
results_log.append(collect_metrics(
    f'Post-Training (Young={THRESHOLD_YOUNG}, Other={THRESHOLD_OTHER})',
    y_gt_test, y_prob_baseline, y_pred_thresh, young_test
))

results_df = pd.DataFrame(results_log).drop_duplicates(subset='Model').set_index('Model')

print("\n--- RESULTS METRICS TABLE ---")
print(results_df.to_string())

# 3.6 Visualisation
model_colors = [C_INK, C_APPROVE, C_BEST, '#8B00FA', C_DENY]
models = results_df.index.tolist()
x      = np.arange(len(models))
bar_w  = 0.55

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C_BG)

ax = axes[0]
bars = ax.bar(x, results_df['ROC-AUC'], color=model_colors, edgecolor='white', width=bar_w)
ax.axhline(results_df.loc['Baseline RF', 'ROC-AUC'], color='grey', linestyle='--', lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('(', '\n(') for m in models], rotation=15, ha='right', fontsize=8)
ax.set_title('ROC-AUC', fontweight='bold', color=C_INK)
ax.set_ylabel('AUC score', color=C_INK)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
ax.grid(False)

ax = axes[1]
bars = ax.bar(x, results_df['DP gap (age)'], color=model_colors, edgecolor='white', width=bar_w)
bank_dp_ref = results_df.loc['Bank (original)', 'DP gap (age)']
ax.axhline(bank_dp_ref, color=C_DENY, linestyle='--', lw=1.5, label=f'Bank level ({bank_dp_ref:.3f})')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('(', '\n(') for m in models], rotation=15, ha='right', fontsize=8)
ax.set_title('Demographic Parity Gap — Age\n(lower = more equal treatment)', fontweight='bold', color=C_INK)
ax.set_ylabel('Absolute gap in Bad-label rates', color=C_INK)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
ax.legend(fontsize=8)
ax.grid(False)

plt.suptitle('Stage 3 — Intervention Comparison (Focus: Age)', fontsize=12, fontweight='bold', y=1.01, color=C_INK)
plt.tight_layout()
plt.savefig('plots/stage3_comparison.png', bbox_inches='tight', dpi=150)
plt.close()

# ── STAGE 4: Feature Importance ───────────────────────────────────────────────
raw_imp = pd.Series(rf_baseline.feature_importances_, index=X_train.columns)

def original_attr(col):
    for key in ATTR_LABEL:
        if col == key or col.startswith(key + '_'):
            return key
    return col

grouped_imp = raw_imp.groupby(raw_imp.index.map(original_attr)).sum().sort_values(ascending=False)
importances_top = grouped_imp[grouped_imp > 0.01].head(15)

demo_feats = {'Attribute13', 'is_young', 'sex_male', 'sex_female',
              'Attribute9', 'marital_status_single', 'marital_status_married/widowed',
              'marital_status_divorced/separated', 'marital_status_divorced/separated/married'}

bar_colors = [C_DENY if f in demo_feats else C_APPROVE for f in importances_top.index]
readable_labels = [readable(f) for f in importances_top.index[::-1]]

fig, ax = plt.subplots(figsize=(11, 6), facecolor=C_BG)
bars = ax.barh(readable_labels, importances_top.values[::-1], color=bar_colors[::-1], edgecolor='white')
ax.set_xlabel('Aggregated MDI share (summed across dummy columns)', fontsize=10, color=C_INK)
ax.set_title('Feature Importance — Baseline Random Forest\n(red = demographic features, blue-violet = financial features)', fontweight='bold', color=C_INK)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
ax.set_xlim(0, importances_top.values.max() * 1.20)
ax.grid(False)

ax.legend(handles=[
    Patch(facecolor=C_APPROVE, label='Financial / behavioural'),
    Patch(facecolor=C_DENY,    label='Demographic (age / sex)'),
], fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('plots/stage4_feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()

# ── STAGE 5: Synthesis and Conclusions ────────────────────────────────────────
all_preds = {
    'Bank (original)'          : bank_pred_binary,
    'Baseline RF'              : y_pred_baseline,
    'Pre-Training (Reweighing)': y_pred_rw,
    'In-Training (Cost 1:5)'   : y_pred_cost,
    f'Post-Training\n(Young={THRESHOLD_YOUNG}, Other={THRESHOLD_OTHER})': y_pred_thresh,
}
model_colors_5 = [C_INK, C_APPROVE, C_BEST, '#8B00FA', C_DENY]

sr_rows = []
for name, preds in all_preds.items():
    approved = (pd.Series(preds, index=y_gt_test.index) == 0).astype(int)
    sr_rows.append({
        'Model'    : name,
        'SR young' : approved[young_test == 1].mean(),
        'SR other' : approved[young_test == 0].mean(),
    })
sr_df = pd.DataFrame(sr_rows)

x     = np.arange(len(sr_df))
bar_w = 0.32

fig, ax = plt.subplots(figsize=(12, 5), facecolor=C_BG)
ax.bar(x - bar_w/2, sr_df['SR young'], width=bar_w, color=model_colors_5, edgecolor='white', alpha=0.95)
ax.bar(x + bar_w/2, sr_df['SR other'], width=bar_w, color=model_colors_5, edgecolor='white', alpha=0.50)

for i in range(len(sr_df)):
    gap = sr_df['SR other'].iloc[i] - sr_df['SR young'].iloc[i]
    top = max(sr_df['SR young'].iloc[i], sr_df['SR other'].iloc[i]) + 0.025
    clr = C_DENY if abs(gap) > 0.05 else C_BEST
    ax.annotate(f'D={gap:+.3f}', xy=(x[i], top), ha='center', va='bottom', fontsize=9, color=clr, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(sr_df['Model'].tolist(), rotation=12, ha='right', fontsize=9)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_ylim(0, 1.15)
ax.set_ylabel('Share not flagged as Bad', fontsize=10, color=C_INK)
ax.set_title('Approval Rate by Age Group — Before and After Interventions', fontweight='bold', color=C_INK)
ax.grid(False)

ax.legend(handles=[
    Patch(facecolor='grey', alpha=0.95, label='Young (Age <= 25)'),
    Patch(facecolor='grey', alpha=0.50, label='Other (Age > 25)'),
], fontsize=9, loc='lower right')

plt.tight_layout()
plt.savefig('plots/stage5_selection_rates.png', bbox_inches='tight', dpi=150)
plt.close()

# Вывод итогов
stage3_results = results_df.copy()
print("\n" + "=" * 55)
print("  FINAL EVALUATION — ALL MODELS")
print("=" * 55)
print(stage3_results.to_string())

bank_dp  = stage3_results.loc['Bank (original)', 'DP gap (age)']
base_dp  = stage3_results.loc['Baseline RF',     'DP gap (age)']

excl_bank = [m for m in stage3_results.index if m != 'Bank (original)']
best_m    = stage3_results.loc[excl_bank, 'DP gap (age)'].idxmin()
best_dp   = stage3_results.loc[best_m, 'DP gap (age)']

print("\nSTAGE 5 — SYNTHESIS AND CONCLUSIONS")
print("=" * 55)
print(f"1. Bank DP gap (age)     : {bank_dp:.4f}")
print(f"   Baseline model DP gap : {base_dp:.4f} (+{base_dp - bank_dp:.4f} vs bank — amplified)")
print(f"2. Best intervention     : {best_m} (DP gap = {best_dp:.4f})")
print("\nExecution complete.")
