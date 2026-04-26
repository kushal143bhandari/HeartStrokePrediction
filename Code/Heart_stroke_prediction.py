# CS 7830-01  - Machine Learning
# Final Project   :  Heart Stroke Prediction
# Group           :  Data Explorers

# Team Members:
#   Bhargav Yendluri  |  UID: U01148249
#   Kushal Bhandari   |  UID: U01137652

# Goal: Use patient health records to predict whether a person is likely to have a stroke (1 = stroke, 0 = no stroke).

# Dataset:
#   Kaggle Stroke Prediction Dataset  -  fedesoriano (2021)
#   5,110 patient records, 11 input features, 4.87% stroke rate
#   Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

# Models:
#   1. Logistic Regression
#   2. Random Forest
#   3. K-Nearest Neighbors (KNN)

# Import Libraries

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from imblearn.over_sampling import SMOTE



def save_and_open(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {filename}")
    try:
        if sys.platform == "win32":
            os.startfile(filename)
        elif sys.platform == "darwin":
            subprocess.call(["open", filename])
        else:
            subprocess.call(["xdg-open", filename])
    except Exception:
        print(f"  (Open {filename} manually to view)")

# LOAD THE DATASET
csv_path = "healthcare-dataset-stroke-data.csv"

data = pd.read_csv(csv_path)

print(f"Shape         : {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Stroke cases  : {data['stroke'].sum()} out of {len(data)} "
      f"({data['stroke'].mean()*100:.2f}%)")
print(f"No stroke     : {(data['stroke']==0).sum()} "
      f"({(data['stroke']==0).mean()*100:.2f}%)")

print("\nColumn data types:")
print(data.dtypes.to_string())

print("\nMissing values per column:")
print(data.isnull().sum().to_string())

print("\nFirst 5 rows:")
print(data.head().to_string())


# EXPLORATORY DATA ANALYSIS(EDA)

print("Exploratory Data Analysis(EDA)")

# Create age groups to make trends easier to see
data["age_group"] = pd.cut(
    data["age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=["Under 18", "18-35", "35-50", "50-65", "Over 65"]
)

print("\nBasic stats for numerical features:")
print(data[["age", "avg_glucose_level", "bmi"]].describe().round(2).to_string())

print("\nStroke rate by age group:")
age_stroke = data.groupby("age_group", observed=True)["stroke"].mean() * 100
for group, rate in age_stroke.items():
    print(f"  {str(group):<10}  :  {rate:.2f}%")

print("\nStroke rate by hypertension status:")
for val, label in [(0, "No hypertension"), (1, "Has hypertension")]:
    rate = data[data["hypertension"] == val]["stroke"].mean() * 100
    print(f"  {label:<22}  :  {rate:.2f}%")

print("\nStroke rate by heart disease status:")
for val, label in [(0, "No heart disease"), (1, "Has heart disease")]:
    rate = data[data["heart_disease"] == val]["stroke"].mean() * 100
    print(f"  {label:<22}  :  {rate:.2f}%")

print("\nStroke rate by smoking status:")
smoke_rates = data.groupby("smoking_status")["stroke"].mean() * 100
for status, rate in smoke_rates.sort_values(ascending=False).items():
    print(f"  {status:<22}  :  {rate:.2f}%")

print("\nCorrelation between each feature and stroke:")
for feat in ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]:
    r = data[feat].corr(data["stroke"])
    print(f"  {feat:<25}  :  r = {r:.4f}")

# EDA visualization
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Heart Stroke Prediction – Exploratory Data Analysis",
             fontsize=16, fontweight="bold")

# Class distribution
ax = axes[0, 0]
counts = data["stroke"].value_counts()
bars   = ax.bar(["No Stroke", "Stroke"], counts.values,
                color=["steelblue", "tomato"], edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 40,
            str(val), ha="center", fontsize=10, fontweight="bold")
ax.set_title("Class Distribution")
ax.set_ylabel("Number of Patients")
ax.spines[["top", "right"]].set_visible(False)

# Age by stroke
ax = axes[0, 1]
ax.hist(data[data["stroke"] == 0]["age"], bins=30,
        alpha=0.65, color="steelblue", label="No Stroke", edgecolor="white")
ax.hist(data[data["stroke"] == 1]["age"], bins=30,
        alpha=0.65, color="tomato", label="Stroke", edgecolor="white")
ax.set_title("Age Distribution by Stroke")
ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# Glucose by stroke
ax = axes[0, 2]
ax.hist(data[data["stroke"] == 0]["avg_glucose_level"], bins=30,
        alpha=0.65, color="steelblue", label="No Stroke", edgecolor="white")
ax.hist(data[data["stroke"] == 1]["avg_glucose_level"], bins=30,
        alpha=0.65, color="tomato", label="Stroke", edgecolor="white")
ax.set_title("Avg Glucose Level by Stroke")
ax.set_xlabel("Glucose (mg/dL)"); ax.set_ylabel("Count"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# BMI by stroke
ax = axes[1, 0]
data_bmi = data.dropna(subset=["bmi"])
ax.hist(data_bmi[data_bmi["stroke"] == 0]["bmi"], bins=30,
        alpha=0.65, color="steelblue", label="No Stroke", edgecolor="white")
ax.hist(data_bmi[data_bmi["stroke"] == 1]["bmi"], bins=30,
        alpha=0.65, color="tomato", label="Stroke", edgecolor="white")
ax.set_title("BMI Distribution by Stroke")
ax.set_xlabel("BMI"); ax.set_ylabel("Count"); ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# Hypertension
ax = axes[1, 1]
hyp = data.groupby("hypertension")["stroke"].mean() * 100
ax.bar(["No Hypertension", "Hypertension"], hyp.values,
       color=["steelblue", "tomato"], edgecolor="white")
ax.set_title("Stroke Rate by Hypertension")
ax.set_ylabel("Stroke Rate (%)")
ax.spines[["top", "right"]].set_visible(False)

# Heart disease
ax = axes[1, 2]
hd = data.groupby("heart_disease")["stroke"].mean() * 100
ax.bar(["No Heart Disease", "Heart Disease"], hd.values,
       color=["steelblue", "tomato"], edgecolor="white")
ax.set_title("Stroke Rate by Heart Disease")
ax.set_ylabel("Stroke Rate (%)")
ax.spines[["top", "right"]].set_visible(False)

# Age group
ax = axes[2, 0]
ax.bar(age_stroke.index, age_stroke.values, color="mediumpurple", edgecolor="white")
ax.set_title("Stroke Rate by Age Group")
ax.set_ylabel("Stroke Rate (%)")
ax.set_xticklabels(age_stroke.index, rotation=20)
ax.spines[["top", "right"]].set_visible(False)

# Smoking status
ax = axes[2, 1]
sm = data.groupby("smoking_status")["stroke"].mean().sort_values() * 100
ax.barh(sm.index, sm.values, color="goldenrod", edgecolor="white")
ax.set_title("Stroke Rate by Smoking Status")
ax.set_xlabel("Stroke Rate (%)")
ax.spines[["top", "right"]].set_visible(False)

# Correlation heatmap
ax = axes[2, 2]
num_df = data[["age", "avg_glucose_level", "bmi",
               "hypertension", "heart_disease", "stroke"]].dropna()
corr = num_df.corr()
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
            mask=np.triu(np.ones_like(corr, dtype=bool)), cbar=True)
ax.set_title("Correlation Heatmap")

save_and_open("eda.png")

# CORRELATION MATRIX
  
from sklearn.preprocessing import LabelEncoder as _LE
 
df_corr = data.copy()
df_corr.drop(columns=["id", "age_group"], errors="ignore", inplace=True)
df_corr = df_corr[df_corr["gender"] != "Other"].reset_index(drop=True)
df_corr["bmi"].fillna(df_corr["bmi"].median(), inplace=True)
 
_le = _LE()
for _col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    df_corr[_col] = _le.fit_transform(df_corr[_col].astype(str))
 
# Rename columns so the heatmap labels are readable
df_corr.rename(columns={
    "avg_glucose_level": "Avg Glucose",
    "heart_disease":     "Heart Disease",
    "hypertension":      "Hypertension",
    "ever_married":      "Ever Married",
    "work_type":         "Work Type",
    "Residence_type":    "Residence",
    "smoking_status":    "Smoking",
    "stroke":            "Stroke (Target)"
}, inplace=True)
 
corr_matrix = df_corr.corr()
 
# Full correlation heatmap
# Only the lower triangle is shown to avoid duplicate information.
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
 
import seaborn as _sns
_cmap = _sns.diverging_palette(220, 10, as_cmap=True)
_sns.heatmap(corr_matrix, ax=ax, mask=mask,
             annot=True, fmt=".2f", annot_kws={"size": 9, "weight": "bold"},
             cmap=_cmap, vmin=-1, vmax=1,
             linewidths=0.5, linecolor="white",
             cbar_kws={"shrink": 0.75, "label": "Pearson r"})
 
ax.set_title("Correlation Matrix  -  All Features",
             fontsize=14, fontweight="bold", pad=14)
ax.tick_params(labelsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
 
save_and_open("correlation_matrix_full.png")
 
# Each feature's individual correlation with stroke 
stroke_corr = corr_matrix["Stroke (Target)"].drop("Stroke (Target)").sort_values()
bar_colors  = ["#EF4444" if v > 0 else "#3B82F6" for v in stroke_corr.values]
 
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(stroke_corr.index, stroke_corr.values,
               color=bar_colors, edgecolor="white", height=0.65)
ax.axvline(0, color="#1F2937", linewidth=1.2)
 
for bar, val in zip(bars, stroke_corr.values):
    xpos = val + 0.006 if val >= 0 else val - 0.006
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha,
            fontsize=10, fontweight="bold", color="#1F2937")
 
ax.set_title("Feature Correlations with Stroke (Target Variable)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Pearson Correlation (r)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
 
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color="#EF4444", label="Positive  (higher value  ->  more stroke risk)"),
    mpatches.Patch(color="#3B82F6", label="Negative  (higher value  ->  less stroke risk)")
]
ax.legend(handles=legend_handles, fontsize=9, loc="lower right",
          frameon=True, framealpha=0.9)
 
save_and_open("stroke_correlations.png")
 
# Print the values to the terminal
print("\n  Correlation with Stroke (Target Variable):")
for feat, r in stroke_corr.sort_values(ascending=False).items():
    print(f"    {feat:<22} : r = {r:+.4f}")

# DATA CLEANING AND PREPROCESSING

print("Data Cleaning and Preprocessing\n")

df = data.copy()

# Drop columns we do not need for training
df.drop(columns=["id", "age_group"], errors="ignore", inplace=True)

# We drop the outlier
df = df[df["gender"] != "Other"].reset_index(drop=True)
print(f"Dropped 1 row where gender = 'Other'.\n Rows remaining: {len(df)}")


bmi_median = df["bmi"].median()
df["bmi"].fillna(bmi_median, inplace=True)
print(f"Filled {data['bmi'].isnull().sum()} missing BMI values "
      f"with the median ({bmi_median:.1f})")

cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
encoder  = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))
print(f"Label encoded columns: {cat_cols}")

# Separate input features from the target variable
X = df.drop("stroke", axis=1)
y = df["stroke"]
feature_names = X.columns.tolist()

print(f"\nInput features ({len(feature_names)}): {feature_names}")
print(f"\nClass counts after cleaning:")
print(f"  Stroke (1)    : {y.sum()}")
print(f"  No stroke (0) : {(y == 0).sum()}")

# Split into training (80%) and test (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain set : {len(X_train)} samples")
print(f"Test set  : {len(X_test)} samples")

scaler         = StandardScaler()
X_train_scaled = np.nan_to_num(scaler.fit_transform(X_train))
X_test_scaled  = np.nan_to_num(scaler.transform(X_test))

# The class imbalance is high
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print(f"\nAfter SMOTE (applied only to training set):")
print(f"  Total training samples : {len(X_train_bal)}")
print(f"  Stroke cases           : {y_train_bal.sum()}")
print(f"  No-stroke cases        : {(y_train_bal == 0).sum()}")


# DEFINE THE THREE MODELS

print(" Defining the Three Models\n")

# Model 1: Logistic Regression:
lr_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    solver="lbfgs",
    random_state=42
)

# Model 2: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Model 3: K-Nearest Neighbors
knn_model = KNeighborsClassifier(
    n_neighbors=7,
    metric="euclidean"
)

models = {
    "Logistic Regression"  : lr_model,
    "Random Forest"        : rf_model,
    "K-Nearest Neighbors"  : knn_model
}

for name in models:
    print(f"  Ready: {name}")


# TRAINING AND EVALUATION

print(" Training and Evaluating All Three Models")
print("  Metrics: Accuracy | Precision | Recall | F1-Score | AUC-ROC")

# We use 5-fold cross-validation on the training
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for model_name, model in models.items():

    print(f"  Training: {model_name}")
    # Train on the SMOTE-balanced training set
    model.fit(X_train_bal, y_train_bal)

    # Predict on the untouched test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Compute all required metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)
    cv_auc    = cross_val_score(model, X_train_bal, y_train_bal,
                                 scoring="roc_auc", cv=cv).mean()

    # Print results
    print(f"\n  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%   - catching real strokes")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {roc_auc*100:.2f}%")
    print(f"  CV-AUC    : {cv_auc*100:.2f}%  (5-fold cross-validation)")

    print(f"\n  Full Classification Report:\n")
    print(classification_report(y_test, y_pred,
          target_names=["No Stroke", "Stroke"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:\n")
    print(f"                  Predicted No    Predicted Yes")
    print(f"  Actual No    :  {tn:>10}      {fp:>10}")
    print(f"  Actual Yes   :  {fn:>10}      {tp:>10}")

    results.append({
        "Model"         : model_name,
        "Accuracy"      : accuracy,
        "Precision"     : precision,
        "Recall"        : recall,
        "F1-Score"      : f1,
        "ROC-AUC"       : roc_auc,
        "CV-AUC"        : cv_auc,
        "predictions"   : y_pred,
        "probabilities" : y_prob
    })


# COMPARISON TABLE

print(" Model Comparison Summary:\n")

summary = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("predictions", "probabilities")}
    for r in results
])

display_df = summary.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV-AUC"]:
    display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

print("\n" + display_df.to_string(index=False))

print("\nBest performing model per metric:")
for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV-AUC"]:
    best_idx   = summary[metric].idxmax()
    best_name  = summary.loc[best_idx, "Model"]
    best_score = summary.loc[best_idx, metric]
    print(f"  {metric:<12} :  {best_name}  ({best_score*100:.2f}%)")


# VISUALIZATIONS

print(" Generating Visualizations:\n\n")

model_colors = {
    "Logistic Regression"  : "#3B82F6",   # blue
    "Random Forest"        : "#10B981",   # green
    "K-Nearest Neighbors"  : "#F59E0B"    # amber
}

# Plot A: All metrics bar chart
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV-AUC"]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Model Comparison – All Metrics", fontsize=15, fontweight="bold")

for ax, metric in zip(axes.flat, metric_names):
    values = [r[metric] for r in results]
    labels = [r["Model"] for r in results]
    colors = [model_colors[r["Model"]] for r in results]
    bars   = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

    # Put a gold border on the best bar for that metric 
    best_i = int(np.argmax(values))
    bars[best_i].set_edgecolor("gold")
    bars[best_i].set_linewidth(3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xticklabels(labels, rotation=12, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

save_and_open("model_comparison.png")

#  Plot B: ROC Curves for all three models on the same graph
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5,
        label="Random baseline  (AUC = 0.50)")

for result, style in zip(results, ["solid", "dashed", "dashdot"]):
    fpr, tpr, _ = roc_curve(y_test, result["probabilities"])
    ax.plot(fpr, tpr, linewidth=2.5, linestyle=style,
            color=model_colors[result["Model"]],
            label=f"{result['Model']}  (AUC = {result['ROC-AUC']:.4f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – All Three Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
save_and_open("roc_curves.png")

# Plot C: Confusion Matrices for all three models side by side
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Confusion Matrices – All Three Models", fontsize=13, fontweight="bold")

for ax, result in zip(axes, results):
    cm = confusion_matrix(y_test, result["predictions"])
    tn, fp, fn, tp = cm.ravel()
    ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=["No Stroke", "Stroke"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"{result['Model']}\nTP={tp}  FP={fp}  FN={fn}  TN={tn}",
        fontsize=10, fontweight="bold"
    )
save_and_open("confusion_matrices.png")

# Plot D: Feature Importance (Random Forest only — KNN has none) 
fig, ax = plt.subplots(figsize=(9, 6))
imp        = models["Random Forest"].feature_importances_
sorted_idx = np.argsort(imp)
bars       = ax.barh([feature_names[i] for i in sorted_idx],
                      imp[sorted_idx], color="#10B981", edgecolor="white")
for bar, val in zip(bars, imp[sorted_idx]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance  –  Random Forest", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.spines[["top", "right"]].set_visible(False)
save_and_open("feature_importance.png")

# Plot E: Radar Chart
categories = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
N          = len(categories)
angles     = [n / float(N) * 2 * np.pi for n in range(N)]
angles.append(angles[0])

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for result in results:
    vals = [result[c] for c in categories] + [result[categories[0]]]
    ax.plot(angles, vals, "o-", linewidth=2,
            color=model_colors[result["Model"]], label=result["Model"])
    ax.fill(angles, vals, alpha=0.08, color=model_colors[result["Model"]])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_title("Radar Chart – 3-Model Overview",
             fontsize=12, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.12), fontsize=10)
save_and_open("radar.png")

# Plot F: Grouped Bar Chart for all metrics side by side
metric_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x           = np.arange(len(metric_list))
bw          = 0.26

fig, ax = plt.subplots(figsize=(12, 6))
for i, result in enumerate(results):
    vals = [result[m] for m in metric_list]
    bars = ax.bar(x + i * bw - bw, vals, bw * 0.9,
                   label=result["Model"],
                   color=model_colors[result["Model"]],
                   edgecolor="white", alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metric_list, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("All Three Models Side by Side", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
save_and_open("grouped_bar.png")



lr  = summary[summary["Model"] == "Logistic Regression"].iloc[0]
rf  = summary[summary["Model"] == "Random Forest"].iloc[0]
knn = summary[summary["Model"] == "K-Nearest Neighbors"].iloc[0]

print(f"""
Results on the real patient test set (1,022 samples):

  Logistic Regression:
    ROC-AUC  = {lr['ROC-AUC']*100:.2f}% 
    Recall   = {lr['Recall']*100:.2f}%  
    Accuracy = {lr['Accuracy']*100:.2f}%
    F1-Score = {lr['F1-Score']*100:.2f}%

  Random Forest:
    Accuracy = {rf['Accuracy']*100:.2f}%  
    CV-AUC   = {rf['CV-AUC']*100:.2f}%  
    Recall   = {rf['Recall']*100:.2f}%
    F1-Score = {rf['F1-Score']*100:.2f}%

  K-Nearest Neighbors:
    Accuracy = {knn['Accuracy']*100:.2f}%
    Recall   = {knn['Recall']*100:.2f}%
    ROC-AUC  = {knn['ROC-AUC']*100:.2f}%
    F1-Score = {knn['F1-Score']*100:.2f}%
""")