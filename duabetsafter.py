#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:06:33 2020

@author: duna55
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:42:30 2020

@author: duna55
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , f1_score ,recall_score , precision_score,accuracy_score ,confusion_matrix ,roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
#import mlxtend
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
hyper = pd.read_csv('export_diabetes2.csv')

print(hyper.head())
print("dimension of Hypertension data: {}".format(hyper.shape))
print(hyper.groupby('Outcome').size())
#hyper=hyper.drop('id',axis=1)


#hyper = ['id','hyper',axis=1]
#hyper.

X=hyper.drop('Outcome',axis=1)
y= hyper['Outcome']


print(hyper.head())




model = PCA(n_components=8).fit(X)
X_pc = model.transform(X)

# number of components
n_pcs= model.components_.shape[0]

# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

initial_feature_names = ['Age','DBP'	,'Obese',	'SBP','bmi'	,'gender'	,'hc'	,'wc'	,'whr']
# get the names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

# LIST COMPREHENSION HERE AGAIN
dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

# build the dataframe
df = pd.DataFrame(dic.items())
print(df)
X_train, X_test, y_train, y_test = train_test_split(X_pc,y,test_size=0.2 ,random_state=42)
print("SVC : ")
svc = SVC(kernel='linear')
model = svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
confusion_matrix(y_test, y_pred)
confusion_matrix(y_test, y_pred)
classi=classification_report(y_test,y_pred)
print(classi)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print('Accuracy: ', accuracy_score(y_test, y_pred),"\nACU",metrics.auc(fpr,tpr))
print("MLP : ")
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500,random_state=42)
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
classi=classification_report(y_test,y_pred)
print(classi)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print('Accuracy: ', accuracy_score(y_test, y_pred),"\nACU",metrics.auc(fpr,tpr))
print("Naive bayes : ")
naive = GaussianNB().fit(X_train, y_train)
y_pred = naive.predict(X_test)
confusion_matrix(y_test, y_pred)
classi=classification_report(y_test,y_pred)
print(classi)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print('Accuracy: ', accuracy_score(y_test, y_pred),"\nACU",metrics.auc(fpr,tpr))
print("Knn : ")
clf = KNeighborsClassifier(n_neighbors=9).fit(X_train, y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)
classi=classification_report(y_test,y_pred)
print(classi)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print('Accuracy: ', accuracy_score(y_test, y_pred),"\nACU",metrics.auc(fpr,tpr))
print("DecisionTree : ")
def run_DecisionTree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=1).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    classi=classification_report(y_test,y_pred)
    print(classi)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print('Accuracy: ', accuracy_score(y_test, y_pred),"\nACU",metrics.auc(fpr,tpr))
get_ipython().run_cell_magic('time', '', 'print("The result of Decision Tree before features selection:")\nrun_DecisionTree(X_train, X_test, y_train, y_test)')
print("Random forest : ")
def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    classi=classification_report(y_test,y_pred)
    print(classi)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    print('Accuracy: ', accuracy_score(y_test, y_pred),"\nACU",metrics.auc(fpr,tpr))
get_ipython().run_cell_magic('time', '', 'print("The result of Ranodm forest before features selection:")\nrun_randomForest(X_train, X_test, y_train, y_test)')

print("Logistic Regression")
logreg = LogisticRegression().fit(X_train, y_train)
y_pred = logreg.predict(X_test)
confusion_matrix(y_test, y_pred)
classi=classification_report(y_test,y_pred)
print(classi)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
print('Accuracy: ', accuracy_score(y_test, y_pred),"\nACU",metrics.auc(fpr,tpr))
