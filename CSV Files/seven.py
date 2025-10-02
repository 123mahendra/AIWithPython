####################Task1############################3

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import metrics
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
#
# df = pd.read_csv('exams.csv', skiprows=0, delimiter=',')
#
# print(df)
#
# X = df.iloc[:, 0:2]
# Y = df.iloc[:, -1]
#
# admit_yes = df.loc[Y == 1]
# admit_no = df.loc[Y == 0]
#
# plt.scatter(admit_no.iloc[:,0], admit_no.iloc[:,1], label='admit no')
# plt.scatter(admit_yes.iloc[:,0], admit_yes.iloc[:,1], label='admit yes')
#
# plt.xlabel('exam1')
# plt.ylabel('exam2')
# plt.legend()
#
# plt.show()
#
# X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=5)
#
# print(X_train.shape)
#
# lm = LogisticRegression()
# lm.fit(X_train, y_train)
# y_train_predict = lm.predict(X_train)
# y_test_predict = lm.predict(X_test)
#
# metrics.ConfusionMatrixDisplay.from_estimator(lm,X_test,y_test)
#
# plt.show()
#
# cnf_matrix = metrics.confusion_matrix(y_test, y_test_predict)
#
# print(cnf_matrix)
#
# print("Accuracy:",metrics.accuracy_score(y_test, y_test_predict))
# print("Precision:",metrics.precision_score(y_test, y_test_predict))
# print("Recall:",metrics.recall_score(y_test, y_test_predict))
#
# TN, FP, FN, TP = cnf_matrix.ravel()
#
# print("True Negatives (TN):", TN)
# print("False Positives (FP):", FP)
# print("False Negatives (FN):", FN)
# print("True Positives (TP):", TP)
#
# new_y_test_predict = lm.predict(X_test)
#
# correct_yes = (new_y_test_predict == 1) & (y_test == 1)
# incorrect_yes = (new_y_test_predict == 1) & (y_test == 0)
# correct_no = (new_y_test_predict == 0) & (y_test == 0)
# incorrect_no = (new_y_test_predict == 0) & (y_test == 1)
#
# plt.scatter(X_test.loc[correct_yes, X_test.columns[0]], X_test.loc[correct_yes, X_test.columns[1]], c="blue", marker='+', label='pred correct yes')
# plt.scatter(X_test.loc[incorrect_yes, X_test.columns[0]], X_test.loc[incorrect_yes, X_test.columns[1]], c="red", marker='o', label='pred incorrect yes')
# plt.scatter(X_test.loc[correct_no, X_test.columns[0]], X_test.loc[correct_no, X_test.columns[1]], c="green", marker='+', label='pred correct no')
# plt.scatter(X_test.loc[incorrect_no, X_test.columns[0]], X_test.loc[incorrect_no, X_test.columns[1]], c="yellow", marker='o', label='pred incorrect no')
#
# plt.xlabel('exam1')
# plt.ylabel('exam2')
# plt.legend()
#
# plt.show()




######################Task2####################################
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
#
# df = pd.read_csv('iris.csv', skiprows=0, delimiter=',')
#
# print(df.head())
#
# X = df.drop("species", axis=1)
# y = df["species"]
#
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
#
# print("Training data shape:", x_train.shape)
# print("Testing data shape:", x_test.shape)
#
#
# log_model = LogisticRegression(max_iter=200)
# log_model.fit(x_train, y_train)
# y_pred_log = log_model.predict(x_test)
#
#
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(x_train, y_train)
# y_pred_knn = classifier.predict(x_test)
#
# print("Logistic Regression Report:")
# print(classification_report(y_test, y_pred_log))
#
#
# print(" Classification Report:")
# print(classification_report(y_test, y_pred_knn))
#
#
# metrics.ConfusionMatrixDisplay.from_estimator(log_model, x_test, y_test)
# plt.title("Logistic Regression Confusion Matrix")
# plt.show()
#
#
# metrics.ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)
# plt.title("Confusion Matrix")
# plt.show()
#
# error = []
#
# for k in range(1, 20):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train, y_train)
#     y_pred_knn = knn.predict(x_test)
#     error.append(np.mean(y_pred_knn != y_test))
#
# plt.plot(range(1, 20), error, marker='o', markersize=10)
#
# plt.xlabel('K')
# plt.ylabel('Mean Error')
#
# plt.show()

####################################################################################################################

#
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import metrics
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay
#
# df = pd.read_csv('iris.csv',delimiter=",")
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)
# ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)
# plt.title("Confusion Matrix")
# plt.show()
#
# print("Classification Report:\n")
# print(classification_report(y_test, y_pred))


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, confusion_matrix
#
# df2 = pd.read_csv("iris.csv")
# print(df2.head())
# X = df2.drop('species', axis=1)
# y = df2['species']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#
# ################Linear Kernel###########################
#
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
# #################Polynomial Kernel#############################
#
# svclassifier = SVC(kernel='poly', degree=2)
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
# ##################RBF##########################################
#
# svclassifier = SVC(kernel='rbf')
# svclassifier.fit(X_train, y_train)
# y_pred = svclassifier.predict(X_test)
#
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

from Assignment5.task1 import y_train

df = pd.read_csv('Admission_Predict.csv', skiprows=0, delimiter=',')
print(df.head())

lm = neighbors.KNeighborsRegressor(n_neighbors=5)

