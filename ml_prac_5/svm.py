import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

# load the dataset
data = pd.read_csv(r"C:/programs/machine-learning-practicals/ml_prac_5/07_loan_approval.csv")

print("Rows and columns:", data.shape)
print(data.head())
print("\nMissing values per column:")
print(data.isnull().sum())
print("\nHow many approved (Y) and not approved (N):")
print(data["LoanApproved"].value_counts())
print()

# data preparation
data = data.drop(columns=["ApplicantID"])

y = data["LoanApproved"].map({"Y": 1, "N": 0})

X = data.drop(columns=["LoanApproved"])

X = pd.get_dummies(X, drop_first=True)

print("Feature columns used:")
print(list(X.columns))
print()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))
print()

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   
X_test_scaled = scaler.transform(X_test)         
print("Done. Each column now has mean 0 and standard deviation 1.")
print()

def evaluate(name, model):
    pred = model.predict(X_test_scaled)
    print(f"{name}")
    print("  Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("  Precision:", round(precision_score(y_test, pred), 4))
    print("  Recall   :", round(recall_score(y_test, pred), 4))
    print("  F1-score :", round(f1_score(y_test, pred), 4))
    print("  Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, pred))
    print()
    return pred, {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    }


# svm with linear kernel
model1 = SVC(kernel="linear", C=1.0)
model1.fit(X_train_scaled, y_train)
pred1, res1 = evaluate("Linear SVM", model1)

# svm with rbf kernel
model2 = SVC(kernel="rbf", C=1.0, gamma="scale")
model2.fit(X_train_scaled, y_train)
pred2, res2 = evaluate("RBF SVM", model2)

# tuning the svm with grid search
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["linear", "rbf"],
}
grid = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

print("Best settings :", grid.best_params_)
print("Best CV score :", round(grid.best_score_, 4))
print()

model3 = grid.best_estimator_
pred3, res3 = evaluate("Tuned SVM", model3)

# results and comparison
results = pd.DataFrame([res1, res2, res3])
print(results.round(4).to_string(index=False))

best = results.loc[results["Accuracy"].idxmax(), "Model"]
print("\nBest model (highest accuracy):", best)
print()

print("Detailed report for the tuned SVM:")
print(classification_report(y_test, pred3, target_names=["Not approved (N)", "Approved (Y)"]))

# plotting
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, pred, title in zip(axes, [pred1, pred2, pred3],
                           ["Linear SVM", "RBF SVM", "Tuned SVM"]):
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["N", "Y"], ax=ax, colorbar=False
    )
    ax.set_title(title)
plt.tight_layout()
plt.show()