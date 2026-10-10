import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Set a random seed for reproducibility
np.random.seed(42)

# ---------------------------
# 1. Data Loading & Preprocessing (same as in Part 1)
# ---------------------------
data = pd.read_csv("dataset.csv", sep=";")
print("First 5 rows of the dataset:")
print(data.head())

# Save a copy of the original features before one-hot encoding (for Distance and Competition)
X_raw = data.drop(columns=["ShotType"])

# Separate target and features
y_raw = data["ShotType"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nLabel mapping for ShotType:")
print(label_mapping)

# One-hot encode categorical features
categorical_cols = ["Competition", "PlayerType", "Movement"]
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)


# ---------------------------
# 2. Define Evaluation Metrics (same as in Part 1)
# ---------------------------
def compute_metrics(y_true, y_pred_proba):
    epsilon = 1e-15
    true_class_probs = np.array([max(p, epsilon) for p in y_pred_proba[np.arange(len(y_true)), y_true]])
    log_score = -np.mean(np.log(true_class_probs))
    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    accuracy = np.mean(y_pred_labels == y_true)
    return log_score, accuracy


# ---------------------------
# 3. Additional Analysis for Part 2: Collect per-sample data for Logistic Regression
# ---------------------------

# We use the same 5-fold cross-validation as before.
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Lists to store per-sample information from all folds
lr_errors = []  # Per-sample log-loss errors
lr_corrects = []  # 1 if prediction correct, 0 if not
lr_distances = []  # Distance value for each test sample
lr_competitions = []  # Competition type for each test sample

fold_index = 1
for train_index, test_index in kf.split(X):
    # Get training and test splits for X (features), y (labels)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train logistic regression on this fold
    lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
    lr_model.fit(X_train, y_train)
    y_pred_proba_lr = lr_model.predict_proba(X_test)

    # For each test sample in this fold, compute per-sample log-loss error and record additional info
    for i, idx in enumerate(test_index):
        true_label = y[idx]
        # Avoid log(0) by ensuring a minimum probability (as in compute_metrics)
        prob_true = max(y_pred_proba_lr[i, true_label], 1e-15)
        sample_error = -np.log(prob_true)
        lr_errors.append(sample_error)
        # Record correctness: 1 if predicted label matches true label, else 0
        pred_label = np.argmax(y_pred_proba_lr[i])
        lr_corrects.append(1 if pred_label == true_label else 0)
        # Record the Distance value from the original data (X_raw)
        lr_distances.append(X_raw.iloc[idx]["Distance"])
        # Record the Competition type from the original data (X_raw)
        lr_competitions.append(X_raw.iloc[idx]["Competition"])

    fold_index += 1

# ---------------------------
# 4. Analysis 1: Error Dependence on Distance
# ---------------------------
# Convert lists to NumPy arrays for easier calculations.
lr_errors = np.array(lr_errors)
lr_distances = np.array(lr_distances)

# Compute the Pearson correlation coefficient between Distance and log-loss error.
corr_coef = np.corrcoef(lr_distances, lr_errors)[0, 1]
print("\nAnalysis 1: Correlation between Distance and Log-Loss Error:")
print(f"Pearson correlation coefficient: {corr_coef:.3f}")

# Optionally, we also do bin the distances and compute average error per bin.
bins = [0, 2, 4, 6, 8, 10]  # Example bin edges (modify as appropriate)
bin_indices = np.digitize(lr_distances, bins)
print("\nAverage Log-Loss Error by Distance Bin:")
for b in np.unique(bin_indices):
    bin_error = lr_errors[bin_indices == b]
    bin_mean_error = np.mean(bin_error)
    print(
        f"Bin {b} (Distance roughly {bins[b - 1]} to {bins[b] if b < len(bins) else 'above'}): Mean Error = {bin_mean_error:.3f}")

# ---------------------------
# 5. Analysis 2: Reweighting Performance by Competition Type
# ---------------------------
# We need to calculate the performance (error and accuracy) for each Competition type, then reweight these results.
# First, we combine our per-sample data into a DataFrame for easier grouping.
analysis_df = pd.DataFrame({
    "Competition": lr_competitions,
    "Error": lr_errors,
    "Correct": lr_corrects
})

# Calculate average log-loss error and accuracy for each Competition type.
competition_summary = analysis_df.groupby("Competition").agg({
    "Error": "mean",
    "Correct": "mean",  # mean of 0/1 gives accuracy
}).reset_index()
print("\nPerformance by Competition Type (Original Distribution):")
print(competition_summary)

# Now, reweight these averages to reflect the true distribution:
# True frequencies: NBA: 0.6, and each of the other four types: 0.1.
# Assume the Competition types in your data include 'NBA' and four other types.
true_freq = {
    "NBA": 0.6,
    # The following keys should match the other competition types in your dataset.
    # For example, if your dataset has: "U14", "EuroLeague", "College", "International", assign each 0.1.
}
# For demonstration, let's assume your dataset has these 5 types:
# (You may adjust these keys to match your actual data.)
true_freq = {
    "NBA": 0.6,
    "U14": 0.1,
    "EuroLeague": 0.1,
    "College": 0.1,
    "International": 0.1
}

# Merge the summary with the true frequencies.
competition_summary["True_Freq"] = competition_summary["Competition"].map(true_freq)

# Remove any rows that don't have a true frequency defined (if any).
competition_summary = competition_summary.dropna(subset=["True_Freq"])

# Now compute the weighted average error and weighted accuracy.
weighted_error = np.sum(competition_summary["Error"] * competition_summary["True_Freq"])
weighted_accuracy = np.sum(competition_summary["Correct"] * competition_summary["True_Freq"])

print("\nReweighted Performance (Assuming True Competition Distribution):")
print(f"Weighted Average Log-Loss Error: {weighted_error:.3f}")
print(f"Weighted Average Accuracy: {weighted_accuracy:.3f}")
