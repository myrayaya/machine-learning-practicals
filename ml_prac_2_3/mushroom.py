import pandas as pd
import numpy as np
from math import log2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load the mushroom dataset
df = pd.read_csv(r"C:\programs\machine-learning-practicals\ml_prac_2\11_mushroom_edibility.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumns: ", df.columns.tolist())

# entropy
def entropy(column):
    counts = column.value_counts()
    probabilities = counts / counts.sum()

    return -(probabilities * np.log2(probabilities)).sum()

print("Entropy of the dataset: ", entropy(df["Class"]))
print("\nClass distribution: ")
print(df["Class"].value_counts())

# remove rows with missing values
df = df.dropna()

# remove ID column
X = df.drop(columns = ["Class", "SampleID"])
y = df["Class"]

# encode categorical features
features_encoders = {}

for column in X.columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column].astype(str))
    features_encoders[column] = encoder

# encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y.astype(str))

print("Features prepared successfully.")
print("Number of features: ", X.shape[1])
print("Number of records: ", X.shape[0])
print("Target classes: ", target_encoder.classes_)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = y)

print("Training records: ", len(X_train))
print("Testing records: ", len(X_test))

# naive bayes classifier
nb_model = CategoricalNB()
nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

print("\nNaive Bayes Classifier Results: ")
print("Accuracy: ", accuracy_score(y_test, nb_predictions))
print("\nClassification Report: ")
print(classification_report(y_test, nb_predictions, labels = np.arange(len(target_encoder.classes_)), target_names = target_encoder.classes_, zero_division = 0))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, nb_predictions))

# decision tree classifier
dt_model = DecisionTreeClassifier(criterion = "entropy", random_state = 42)
dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

print("Decision Tree Classifier Results: ")
print("Accuracy: ", accuracy_score(y_test, dt_predictions))
print("\nClassification report: ")
print(classification_report(y_test, dt_predictions, labels = np.arange(len(target_encoder.classes_)), target_names = target_encoder.classes_, zero_division = 0))

print("\nConfusion Matrix: ")
print(confusion_matrix(y_test, dt_predictions))

# model comparison
print("\nModel Comparison: ")
print("-" * 35)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_predictions):2%}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_predictions):2%}")

# select one mushroom from the test dataset
sample = X_test.iloc[[0]]

nb_result = target_encoder.inverse_transform(nb_model.predict(sample))[0]

dt_result = target_encoder.inverse_transform(dt_model.predict(sample))[0]

actual_result = target_encoder.inverse_transform(y_test[:1])[0]

print("Mushroom characteristics:")
print(df.loc[sample.index[0], X.columns])

print("\nActual class:", actual_result)
print("Naive Bayes prediction:", nb_result)
print("Decision Tree prediction:", dt_result)
