# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:40:36 2018

@author: hecha
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC



## read wine dataset

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)


df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

#df_wine.head()


#print pairplot
sns.pairplot(df_wine, size=2.5)
plt.tight_layout()
plt.show()


#print the heatmap
cm = np.corrcoef(df_wine.values.T)
#sns.set(font_scale=1.5)
fig, ax = plt.subplots(figsize=(12,12)) # 控制热力图大小 （size of map）
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2g',
                 annot_kws={'size': 10},
                 ax = ax,
                 yticklabels=df_wine.columns,
                 xticklabels=df_wine.columns)

plt.tight_layout()
plt.show()


# split the data
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                     stratify=y,
                     random_state=42)


#Standardizing the data
sc = StandardScaler() #标准化后为何变为两列？？？
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# Logistic Regression
tuned_parameters = [{'C':np.arange(0.01,1.0,0.01).tolist(),
                     'multi_class':['ovr']}]
clf=GridSearchCV(LogisticRegression(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train,y_train)


lr = clf.best_estimator_
lr. fit(X_train,y_train)

y_train_pred = lr.predict(X_train)
print('Accruacy of Logistic Regression model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = lr.predict(X_test)
print('Accruacy of Logistic Regression model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))
print("")


# SVM
tuned_parameters = {'gamma':np.arange(0.01,1.0,0.01).tolist()}
clf=GridSearchCV(SVC(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train,y_train)

svm = clf.best_estimator_
svm.fit(X_train,y_train)

y_train_pred = svm.predict(X_train)
print('Accruacy of SVM model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = svm.predict(X_test)
print('Accruacy of SVM model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))
print("")


# pca 
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
#print(pca.explained_variance_ratio_.sum())   #用来看pca选中的总方差占比


# LR after PCA
tuned_parameters = [{'C':np.arange(0.01,1.0,0.01).tolist(),
                     'multi_class':['ovr']}]
clf=GridSearchCV(LogisticRegression(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train_pca,y_train)


lr_pca = clf.best_estimator_
lr_pca. fit(X_train_pca,y_train)

y_train_pred = lr_pca.predict(X_train_pca)
print('Accruacy of PCA SVM model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = lr_pca.predict(X_test_pca)
print('Accruacy of PCA SVM model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))
print("")


# SVM after PCA

tuned_parameters = {'gamma':np.arange(0.01,1.0,0.01).tolist()}
clf=GridSearchCV(SVC(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train_pca,y_train)

svm_pca = clf.best_estimator_
svm_pca.fit(X_train_pca, y_train)

y_train_pred = svm_pca.predict(X_train_pca)
print('Accruacy of PCA SVM model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = svm_pca.predict(X_test_pca)
print('Accruacy of PCA SVM model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))

# LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)


# LR after LDA
tuned_parameters = [{'C':np.arange(0.01,1.0,0.01).tolist(),
                     'multi_class':['ovr']}]
clf=GridSearchCV(LogisticRegression(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train_lda,y_train)


lr_lda = clf.best_estimator_
lr_lda. fit(X_train_lda,y_train)

y_train_pred = lr_lda.predict(X_train_lda)
print('Accruacy of LDA Logistic Regression model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = lr_lda.predict(X_test_lda)
print('Accruacy of LDA Logistic Regression model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))
print("")

# SVM after LDA

tuned_parameters = {'gamma':np.arange(0.01,1.0,0.01).tolist()}
clf=GridSearchCV(SVC(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train_lda,y_train)

svm_lda = clf.best_estimator_
svm_lda.fit(X_train_lda, y_train)

y_train_pred = svm_lda.predict(X_train_lda)
print('Accruacy of LDA SVM model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = svm_lda.predict(X_test_lda)
print('Accruacy of LDA SVM model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))
print("")

#kpca
kpca = KernelPCA(n_components = 10, kernel = 'rbf')
X_train_kpca = kpca.fit_transform(X_train_std, y_train)
X_test_kpca = kpca.transform(X_test_std)

# LR after kpca
tuned_parameters = [{'C':np.arange(0.01,1.0,0.01).tolist(),
                     'multi_class':['ovr']}]
clf=GridSearchCV(LogisticRegression(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train_kpca,y_train)


lr_kpca = clf.best_estimator_
lr_kpca. fit(X_train_kpca,y_train)

y_train_pred = lr_kpca.predict(X_train_kpca)
print('Accruacy of kPCA Logistic Regression model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = lr_kpca.predict(X_test_kpca)
print('Accruacy of kPCA Logistic Regression model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))
print("")


# SVM after kpca
tuned_parameters = {'gamma':np.arange(0.01,1.0,0.01).tolist()}
clf=GridSearchCV(SVC(),tuned_parameters,scoring='accuracy',cv=5) 

clf.fit(X_train_kpca,y_train)

svm_kpca = clf.best_estimator_
svm_kpca.fit(X_train_kpca, y_train)

y_train_pred = svm_kpca.predict(X_train_kpca)
print('Accruacy of kPCA SVM model(training set):')
print(metrics.accuracy_score(y_train, y_train_pred))

y_test_pred = svm_kpca.predict(X_test_kpca)
print('Accruacy of kPCA SVM model(testing set):')
print(metrics.accuracy_score(y_test, y_test_pred))
print("")

print("My name is {Chaozhen He}")
print("My NetID is: {che19}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")