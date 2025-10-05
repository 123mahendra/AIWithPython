
#####Banking System using AI with Python#####

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('bank.csv', delimiter=';')

print(df.head())
print(df.dtypes)

df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]

print(df2.head())

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])

print(df3.head())

if df3['y'].dtype == 'object':
    df3['y'] = df3['y'].map({'no':0, 'yes':1})

plt.figure(figsize=(14, 10))
sns.heatmap(df3.corr(), annot=True)
plt.title("Heatmap of Correlation Coefficients")
plt.tight_layout()
plt.show()

corr_with_y = df3.corr()['y'].abs().sort_values(ascending=False)
print("Top 5 variables most correlated with target (by absolute value):")
print(corr_with_y.head(5))

"""

Top 5 variables most correlated with target (by absolute value):
y                   1.000000
poutcome_success    0.283481
poutcome_unknown    0.162038
housing_yes         0.104683
housing_no          0.104683

"""

y = df3['y']
X = df3.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Logistic Regression:")

lm = LogisticRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n {cnf_matrix}")

acc = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc}")

metrics.ConfusionMatrixDisplay.from_estimator(lm, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("K-Nearest Neighbors (K=3):")

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred_knn = classifier.predict(X_test)

print("Classification Report:")
print(metrics.classification_report(y_test, y_pred_knn))

confusion_matrix_knn = metrics.confusion_matrix(y_test, y_pred_knn)
print(f"\nConfusion Matrix:\n {confusion_matrix_knn}")

acc_knn = metrics.accuracy_score(y_test, y_pred_knn)
print(f"\nAccuracy: {acc_knn}")

metrics.ConfusionMatrixDisplay.from_estimator(classifier,X_test,y_test)
plt.title("k-nearest neighbors Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


"""

Findings:
    Logistic Regression achieved around 90% accuracy where as KNN (K=3) achieved around 86% accuracy. 
    So, in this case Logistic Regression perform more accuracy then KNN model in this dataset.

"""
