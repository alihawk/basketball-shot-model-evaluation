# == == == Part 1 and Part 2 Combined Code with Graphs == == ==
# This code performs model evaluation and additional analyses on a basketball shot dataset.
# It compares four models using 5-fold cross-validation and then conducts further analyses
# on how shot Distance relates to prediction error and how model performance changes when
# reweighting Competition frequencies to match real-world data.

# Import necessary libraries for data manipulation, numerical operations, ML models, CV, and plotting.
import pandas as pd  # Data manipulation and CSV input/output.
import numpy as np  # Numerical operations (e.g., mean, std, array operations).
import math  # Mathematical functions (e.g., logarithm).
from sklearn.linear_model import LogisticRegression  # For logistic regression model.
from sklearn.neighbors import KNeighborsClassifier  # For k-Nearest Neighbors model.
from sklearn.model_selection import KFold  # For cross-validation splitting.
from sklearn.preprocessing import LabelEncoder  # For converting categorical target labels to numbers.
import matplotlib.pyplot as plt  # For generating plots.

# Set a random seed for reproducibility of results.
np.random.seed(42)

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
# Load the dataset from a CSV file with ";" as the separator.
data = pd.read_csv("dataset.csv", sep=";")
print("First 5 rows of the dataset:")
print(data.head())

# Separate the target variable (ShotType) and features.
y_raw = data["ShotType"]  # Target column.
X_raw = data.drop(columns=["ShotType"])  # All columns except target.

# Encode target labels (e.g., 'above head', 'layup') to numeric values using LabelEncoder.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
# Create a mapping of original labels to encoded numbers for interpretation.
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nLabel mapping for ShotType:")
print(label_mapping)

# One-hot encode categorical features (Competition, PlayerType, Movement).
categorical_cols = ["Competition", "PlayerType", "Movement"]
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)


# Now X contains only numerical columns.

# ---------------------------
# 2. Define Evaluation Metrics
# ---------------------------
def compute_metrics(y_true, y_pred_proba):
    """
    Compute two evaluation metrics:
      - Log-Score: Negative log likelihood, which penalizes confident but wrong predictions.
      - Accuracy: Proportion of correct predictions.
    Parameters:
      y_true: True labels (NumPy array of integers).
      y_pred_proba: Predicted probabilities (NumPy array of shape [n_samples, n_classes]).
    Returns:
      log_score: The negative log likelihood (lower is better).
      accuracy: The fraction of correct predictions.
    """
    epsilon = 1e-15  # Small constant to avoid taking the logarithm of 0.
    # Extract the probability assigned to the true class for each sample.
    true_class_probs = np.array([max(p, epsilon) for p in y_pred_proba[np.arange(len(y_true)), y_true]])
    # Calculate the log-score (average of -log(true_class_probabilities)).
    log_score = -np.mean(np.log(true_class_probs))
    # For accuracy, predict the class with the highest probability.
    y_pred_labels = np.argmax(y_pred_proba, axis=1)
    accuracy = np.mean(y_pred_labels == y_true)
    return log_score, accuracy


# ---------------------------
# 3. Cross-Validation Setup and Model Evaluation (Part 1)
# ---------------------------
# Set up 5-fold cross-validation.
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize a dictionary to store performance metrics for each model.
results = {
    "Baseline": {"log_score": [], "accuracy": []},
    "LogisticRegression": {"log_score": [], "accuracy": []},
    "kNN_train_fold": {"log_score": [], "accuracy": []},
    "kNN_nested": {"log_score": [], "accuracy": []}
}

# Define candidate k values for the kNN model.
k_candidates = [1, 3, 5, 7]

# Also collect per-sample data for Logistic Regression (assumed best) for further analysis.
lr_errors = []  # Stores log-loss error for each test sample.
lr_corrects = []  # Stores correctness (1 if correct, else 0) for each test sample.
lr_distances = []  # Stores the shot Distance from the original dataset for each test sample.
lr_competitions = []  # Stores the Competition type from the original dataset for each test sample.

fold_index = 1
for train_index, test_index in kf.split(X):
    print(f"\n--- Fold {fold_index} ---")
    # Split data into training and test sets for current fold using integer indexing.
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # ---------------------------
    # 4.1 Baseline Classifier
    # ---------------------------
    # Compute class frequencies from training data.
    unique, counts = np.unique(y_train, return_counts=True)
    class_freq = np.zeros(len(label_encoder.classes_))
    class_freq[unique] = counts / counts.sum()  # Relative frequency for each class.
    # For each test sample, the predicted probability is the same (class frequencies).
    y_pred_proba_baseline = np.tile(class_freq, (len(y_test), 1))
    # Compute metrics for the baseline.
    log_score_baseline, acc_baseline = compute_metrics(y_test, y_pred_proba_baseline)
    results["Baseline"]["log_score"].append(log_score_baseline)
    results["Baseline"]["accuracy"].append(acc_baseline)

    # ---------------------------
    # 4.2 Logistic Regression
    # ---------------------------
    # Initialize and train Logistic Regression.
    lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    lr_model.fit(X_train, y_train)
    # Predict probabilities on the test set.
    y_pred_proba_lr = lr_model.predict_proba(X_test)
    # Compute metrics.
    log_score_lr, acc_lr = compute_metrics(y_test, y_pred_proba_lr)
    results["LogisticRegression"]["log_score"].append(log_score_lr)
    results["LogisticRegression"]["accuracy"].append(acc_lr)

    # Collect per-sample data from Logistic Regression for further analysis.
    for i, idx in enumerate(test_index):
        true_label = y[idx]
        # Ensure probability is not zero.
        prob_true = max(y_pred_proba_lr[i, true_label], 1e-15)
        sample_error = -np.log(prob_true)
        lr_errors.append(sample_error)
        pred_label = np.argmax(y_pred_proba_lr[i])
        lr_corrects.append(1 if pred_label == true_label else 0)
        # Retrieve shot Distance and Competition from the original data.
        lr_distances.append(X_raw.iloc[idx]["Distance"])
        lr_competitions.append(X_raw.iloc[idx]["Competition"])

    # ---------------------------
    # 4.3 kNN with Training-Fold Hyperparameter Optimization
    # ---------------------------
    best_k = None
    best_train_log_score = float("inf")
    # Iterate over candidate values for k.
    for k in k_candidates:
        knn = KNeighborsClassifier(n_neighbors=k)
        # Train on the training set and predict on the training set.
        y_pred_proba_train = knn.fit(X_train, y_train).predict_proba(X_train)
        train_log_score, _ = compute_metrics(y_train, y_pred_proba_train)
        # Select k that minimizes training log-score.
        if train_log_score < best_train_log_score:
            best_train_log_score = train_log_score
            best_k = k
    # Train kNN using the best k on the entire training set.
    knn_best = KNeighborsClassifier(n_neighbors=best_k)
    knn_best.fit(X_train, y_train)
    y_pred_proba_knn_train = knn_best.predict_proba(X_test)
    log_score_knn_train, acc_knn_train = compute_metrics(y_test, y_pred_proba_knn_train)
    results["kNN_train_fold"]["log_score"].append(log_score_knn_train)
    results["kNN_train_fold"]["accuracy"].append(acc_knn_train)

    # ---------------------------
    # 4.4 kNN with Nested Cross-Validation for Hyperparameter Tuning
    # ---------------------------
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=42)
    # Dictionary to store inner CV scores for each candidate k.
    k_scores = {k: [] for k in k_candidates}
    for inner_train_index, inner_val_index in inner_kf.split(X_train):
        X_inner_train, X_inner_val = X_train.iloc[inner_train_index], X_train.iloc[inner_val_index]
        y_inner_train, y_inner_val = y_train[inner_train_index], y_train[inner_val_index]
        # Evaluate each candidate k on inner folds.
        for k in k_candidates:
            knn_inner = KNeighborsClassifier(n_neighbors=k)
            knn_inner.fit(X_inner_train, y_inner_train)
            y_inner_pred_proba = knn_inner.predict_proba(X_inner_val)
            inner_log_score, _ = compute_metrics(y_inner_val, y_inner_pred_proba)
            k_scores[k].append(inner_log_score)
    # Compute average inner log-score for each k and select the best.
    avg_k_scores = {k: np.mean(scores) for k, scores in k_scores.items()}
    best_k_nested = min(avg_k_scores, key=avg_k_scores.get)
    knn_nested = KNeighborsClassifier(n_neighbors=best_k_nested)
    knn_nested.fit(X_train, y_train)
    y_pred_proba_knn_nested = knn_nested.predict_proba(X_test)
    log_score_knn_nested, acc_knn_nested = compute_metrics(y_test, y_pred_proba_knn_nested)
    results["kNN_nested"]["log_score"].append(log_score_knn_nested)
    results["kNN_nested"]["accuracy"].append(acc_knn_nested)

    # Print fold results.
    print(f"Baseline: log_score={log_score_baseline:.3f}, accuracy={acc_baseline:.3f}")
    print(f"Logistic Regression: log_score={log_score_lr:.3f}, accuracy={acc_lr:.3f}")
    print(
        f"kNN (train-fold tuning, best k={best_k}): log_score={log_score_knn_train:.3f}, accuracy={acc_knn_train:.3f}")
    print(
        f"kNN (nested CV, best k={best_k_nested}): log_score={log_score_knn_nested:.3f}, accuracy={acc_knn_nested:.3f}")

    fold_index += 1

# ---------------------------
# 5. Aggregate and Report Overall Results with Uncertainty
# ---------------------------
print("\n=== Overall Cross-Validation Results ===")
for model_name, metrics in results.items():
    avg_log_score = np.mean(metrics["log_score"])
    std_log_score = np.std(metrics["log_score"])
    avg_accuracy = np.mean(metrics["accuracy"])
    std_accuracy = np.std(metrics["accuracy"])
    print(f"{model_name}: Average Log-Score = {avg_log_score:.3f} ± {std_log_score:.3f}, "
          f"Average Accuracy = {avg_accuracy:.3f} ± {std_accuracy:.3f}")

# ---------------------------
# Part 2 Analysis: Using Logistic Regression Data (Assumed Best Model)
# ---------------------------
# Analysis 1: Correlation between Distance and Log-Loss Error
lr_errors = np.array(lr_errors)
lr_distances = np.array(lr_distances)
corr_coef = np.corrcoef(lr_distances, lr_errors)[0, 1]
print("\nAnalysis 1: Correlation between Distance and Log-Loss Error (Logistic Regression):")
print(f"Pearson correlation coefficient: {corr_coef:.3f}")

# Plot the correlation: Scatter plot with custom width and height.
plt.figure(figsize=(8, 6))
plt.scatter(lr_distances, lr_errors, color='purple', alpha=0.6)
plt.title('Correlation between Distance and Log-Loss Error (Logistic Regression)')
plt.xlabel('Distance')
plt.ylabel('Log-Loss Error')
plt.savefig('distance_error_correlation.svg', format='svg')
plt.close()

# Analysis 1: Bin distances and compute average error per bin (Logistic Regression)
bins = [0, 2, 4, 6, 8, 10]
bin_indices = np.digitize(lr_distances, bins)
unique_bins = np.unique(bin_indices)
bin_means = []
bin_labels = []
for b in unique_bins:
    mean_err = np.mean(lr_errors[bin_indices == b])
    bin_means.append(mean_err)
    label = f"{bins[b - 1]}-{bins[b] if b < len(bins) else '>'}"
    bin_labels.append(label)
plt.figure(figsize=(8, 6))
plt.bar(bin_labels, bin_means, color='orange')
plt.title('Average Log-Loss Error by Distance Bin (Logistic Regression)')
plt.xlabel('Distance Bin')
plt.ylabel('Mean Log-Loss Error')
plt.savefig('distance_bin_error.svg', format='svg')
plt.close()

# Analysis 2: Reweighting Performance by Competition Type
analysis_df = pd.DataFrame({
    "Competition": lr_competitions,
    "Error": lr_errors,
    "Correct": lr_corrects
})
competition_summary = analysis_df.groupby("Competition").agg({
    "Error": "mean",
    "Correct": "mean"
}).reset_index()
print("\nPerformance by Competition Type (Original, Logistic Regression):")
print(competition_summary)

# Define true frequencies for competition types.
true_freq = {
    "NBA": 0.6,
    "U14": 0.1,
    "EuroLeague": 0.1,
    "College": 0.1,
    "International": 0.1
}
# Map true frequencies and remove unmatched rows.
competition_summary["True_Freq"] = competition_summary["Competition"].map(true_freq)
competition_summary = competition_summary.dropna(subset=["True_Freq"])
weighted_error = np.sum(competition_summary["Error"] * competition_summary["True_Freq"])
weighted_accuracy = np.sum(competition_summary["Correct"] * competition_summary["True_Freq"])
print("\nReweighted Performance (Assuming True Competition Distribution, Logistic Regression):")
print(f"Weighted Average Log-Loss Error: {weighted_error:.3f}")
print(f"Weighted Average Accuracy: {weighted_accuracy:.3f}")

# Plot reweighted performance: Grouped bar chart.
x = np.arange(len(competition_summary["Competition"]))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width / 2, competition_summary["Error"], width, label='Error', color='salmon')
rects2 = ax.bar(x + width / 2, competition_summary["Correct"], width, label='Accuracy', color='mediumseagreen')
ax.set_ylabel('Performance')
ax.set_title('Performance by Competition (Original Distribution)')
ax.set_xticks(x)
ax.set_xticklabels(competition_summary["Competition"])
ax.legend()
plt.tight_layout()
plt.savefig('competition_performance.svg', format='svg')
plt.close()

# Plot Overall Performance with Uncertainty (for all models)
model_names = list(results.keys())
avg_log_scores = [np.mean(results[model]["log_score"]) for model in model_names]
std_log_scores = [np.std(results[model]["log_score"]) for model in model_names]
avg_accuracies = [np.mean(results[model]["accuracy"]) for model in model_names]
std_accuracies = [np.std(results[model]["accuracy"]) for model in model_names]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(model_names, avg_log_scores, yerr=std_log_scores, capsize=5, color='skyblue')
ax[0].set_title('Overall Log-Score')
ax[0].set_ylabel('Log-Score (Lower is better)')
ax[1].bar(model_names, avg_accuracies, yerr=std_accuracies, capsize=5, color='lightgreen')
ax[1].set_title('Overall Accuracy')
ax[1].set_ylabel('Accuracy (Higher is better)')
plt.tight_layout()
plt.savefig('overall_performance.svg', format='svg')
plt.close()

print("All graphs have been saved as SVG files.")
