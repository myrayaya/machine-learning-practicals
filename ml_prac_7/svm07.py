import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# load the dataset
df = pd.read_csv(
    r"C:\programs\machine-learning-practicals\ml_prac_7\data_banknote_authentication (1).txt",
    header=None
)

df.columns = [
    'Variance',
    'Skewness',
    'Curtosis',
    'Entropy',
    'Class'
]

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df['Class'].value_counts())

print("\nStatistical Summary:")
print(df.describe())


# 3. feature and target selection

X = df.drop('Class', axis=1)
y = df['Class']

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Column:")
print(y.name)


# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Rows:", len(X_train))
print("Testing Rows:", len(X_test))


# feature scaling with linear svm
linear_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear'))
])

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

print("\nLinear SVM")

print("Accuracy :", round(accuracy_score(y_test, y_pred_linear), 4))
print("Precision:", round(precision_score(y_test, y_pred_linear), 4))
print("Recall   :", round(recall_score(y_test, y_pred_linear), 4))
print("F1-score :", round(f1_score(y_test, y_pred_linear), 4))

cm_linear = confusion_matrix(y_test, y_pred_linear)

print("\nConfusion Matrix:")
print(cm_linear)


# 6. feature scaling with rbf svm
rbf_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])

rbf_model.fit(X_train, y_train)

y_pred_rbf = rbf_model.predict(X_test)

print("\nRBF SVM")

print("Accuracy :", round(accuracy_score(y_test, y_pred_rbf), 4))
print("Precision:", round(precision_score(y_test, y_pred_rbf), 4))
print("Recall   :", round(recall_score(y_test, y_pred_rbf), 4))
print("F1-score :", round(f1_score(y_test, y_pred_rbf), 4))

cm_rbf = confusion_matrix(y_test, y_pred_rbf)

print("\nConfusion Matrix:")
print(cm_rbf)


# hyperparameter tuning using grid search
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'svm__kernel': ['rbf']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Settings:", grid_search.best_params_)
print("Best CV Score:", round(grid_search.best_score_, 4))


# evaluation of tuned svm
y_pred_tuned = best_model.predict(X_test)

print("\nTuned SVM")

print("Accuracy :", round(accuracy_score(y_test, y_pred_tuned), 4))
print("Precision:", round(precision_score(y_test, y_pred_tuned), 4))
print("Recall   :", round(recall_score(y_test, y_pred_tuned), 4))
print("F1-score :", round(f1_score(y_test, y_pred_tuned), 4))

cm_tuned = confusion_matrix(y_test, y_pred_tuned)

print("\nConfusion Matrix:")
print(cm_tuned)


# results and comparison
results = pd.DataFrame({
    'Model': ['Linear SVM', 'RBF SVM', 'Tuned SVM'],

    'Accuracy': [
        accuracy_score(y_test, y_pred_linear),
        accuracy_score(y_test, y_pred_rbf),
        accuracy_score(y_test, y_pred_tuned)
    ],

    'Precision': [
        precision_score(y_test, y_pred_linear),
        precision_score(y_test, y_pred_rbf),
        precision_score(y_test, y_pred_tuned)
    ],

    'Recall': [
        recall_score(y_test, y_pred_linear),
        recall_score(y_test, y_pred_rbf),
        recall_score(y_test, y_pred_tuned)
    ],

    'F1': [
        f1_score(y_test, y_pred_linear),
        f1_score(y_test, y_pred_rbf),
        f1_score(y_test, y_pred_tuned)
    ]
})

print("\nModel Comparison:")
print(results.to_string(index=False))

best_model_name = results.loc[
    results['Accuracy'].idxmax(), 'Model'
]

print("\nBest Model (Highest Accuracy):", best_model_name)


# plotting
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

confusion_matrices = [
    cm_linear,
    cm_rbf,
    cm_tuned
]

model_names = [
    'Linear SVM',
    'RBF SVM',
    'Tuned SVM'
]

for ax, cm, name in zip(axes, confusion_matrices, model_names):

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='viridis',
        xticklabels=['Class 0', 'Class 1'],
        yticklabels=['Class 0', 'Class 1'],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.show()


# classification report
print("\nDetailed Report for the Tuned SVM:")

print(
    classification_report(
        y_test,
        y_pred_tuned,
        target_names=['Class 0', 'Class 1']
    )
)


#model comparison results
results.to_csv(
    "SVM_Model_Comparison.csv",
    index=False
)