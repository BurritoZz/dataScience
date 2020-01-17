#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from sklearn.kernel_ridge import KernelRidge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

## Data Preperation (functions):
def ageToInt(ageList):
    res = []
    for age in ageList:
        if (age == '>=80'):
            age = "80"
        ageRange = age.split('-')
        res.append(int(ageRange[0]));
    return res

def replaceZeroMostCommon(x):
    if pd.isnull(x):
        return 0
    else:
        return x

def replaceOneMostCommon(x):
    if pd.isnull(x):
        return 1
    else:
        return x

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

data = pd.read_csv('Data.csv', sep=';')

## Data preparation:
data['Age'] = ageToInt(data['Age'])
data['Fever'] = data['Fever'].map(replaceZeroMostCommon)
data['Duration_of_pain'] = imputer.fit_transform(data[['Duration_of_pain']])
data = data.drop(columns=['Workoverload'])
data['Extremely_nervous'] = imputer.fit_transform(data[['Extremely_nervous']])
data['Relationship_with_colleagues'] = imputer.fit_transform(data[['Relationship_with_colleagues']]) # Misschien deleten
data['Irrational_thoughts_risk_lasting'] = imputer.fit_transform(data[['Irrational_thoughts_risk_lasting']])
data['Irrational_thoughts_work'] = imputer.fit_transform(data[['Irrational_thoughts_work']])
data['Coping_strategy'] = imputer.fit_transform(data[['Coping_strategy']])
data['Kinesiophobia_physical_exercise'] = imputer.fit_transform(data[['Kinesiophobia_physical_exercise']])
data['Kinesiophobia_pain_stop'] = imputer.fit_transform(data[['Kinesiophobia_pain_stop']])
data['Uses_corticosteroids'] = data['Uses_corticosteroids'].map(replaceZeroMostCommon)
data['Serious_disease'] = data['Serious_disease'].map(replaceZeroMostCommon)
data['Weightloss_per_year'] = imputer.fit_transform(data[['Weightloss_per_year']])
data['Loss_muscle_strength'] = data['Loss_muscle_strength'].map(replaceOneMostCommon)
data['Trauma'] = data['Trauma'].map(replaceZeroMostCommon) # Misschien deleten
data['Incoordination'] = data['Incoordination'].map(replaceZeroMostCommon)
data = data.drop(columns='working_ability')


#print(data['Trauma'].value_counts())
#print(data['Trauma'].count())
#print(data['Relationship_with_colleagues'].value_counts())
#print(data['Relationship_with_colleagues'].count())
#print(data)


## Select features for feature selection
#X = data.iloc[:,1:34]
#y = data.iloc[:,0]

#bestfeatures = SelectKBest(score_func=chi2, k=10)
#fit = bestfeatures.fit(X, y)
#dfscores = pd.DataFrame(fit.scores_)
#dfcolumns = pd.DataFrame(X.columns)
#
#featureScores = pd.concat([dfcolumns, dfscores], axis=1)
#featureScores.columns = ['Specs', 'Score']
#print(featureScores.nlargest(10, 'Score'))
#
#from sklearn.ensemble import ExtraTreesClassifier
#import matplotlib.pyplot as plt
#model = ExtraTreesClassifier()
#model.fit(X,y)
#print(model.feature_importances_)
#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#feat_importances.nlargest(10).plot(kind='barh')
#plt.tight_layout()
#plt.savefig('featureImportance')
#
#corrmat = data.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(40,40))
#g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.savefig('hetejongen')


## Classification (with feature selection):
print('With feature selection')
X = data[['Age', 'leg_left_pain_intensity', 'leg_right_pain_intensity', 'arm_right_pain_intensity', 'Decreased_mobility', 'neck_pain_intensity', 'Irrational_thoughts_work', 'Fever', 'Serious_disease', 'Uses_corticosteroids', 'Irrational_thoughts_risk_lasting', 'Coping_strategy', 'Relationship_with_colleagues', 'low_back_pain_intensity', 'Extremely_nervous', 'Kinesiophobia_pain_stop', 'Weightloss_per_year']]
y = data['Treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Ridge')
lin_reg = linear_model.RidgeClassifier()
lin_reg.fit(X_train, y_train)
y_test_pred = lin_reg.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(lin_reg.score(X_test, y_test))

print('SGD')
sgdClassifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
sgdClassifier.fit(X_train,y_train)
y_test_pred = sgdClassifier.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(sgdClassifier.score(X_test, y_test))

print('Linear Discriminant Analysis svd')
linDisc = LinearDiscriminantAnalysis(solver='svd')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(linDisc.score(X_test, y_test))

print('Linear Discriminant Analysis lsqr')
linDisc = LinearDiscriminantAnalysis(solver='lsqr')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(linDisc.score(X_test, y_test))

print('Linear Discriminant Analysis eigen')
linDisc = LinearDiscriminantAnalysis(solver='eigen')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(linDisc.score(X_test, y_test))

print('Quadratic Discriminant Analysis')
quadDisc = QuadraticDiscriminantAnalysis()
quadDisc.fit(X_train, y_train)
y_test_pred = quadDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(quadDisc.score(X_test, y_test))

print('Kernel Ridge Regression')
kerRid = KernelRidge(alpha=1.0)
kerRid.fit(X_train, y_train)
y_test_pred = kerRid.predict(X_test)
y_test_pred = [int(round(x)) for x in y_test_pred]
print(confusion_matrix(y_test, y_test_pred))
print(kerRid.score(X_test, y_test))

print('SVC')
svc = svm.SVC()
svc.fit(X_train, y_train)
y_test_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(svc.score(X_test, y_test))

print('K Nearest Neighbors')
kNeigh = KNeighborsClassifier(n_neighbors=3)
kNeigh.fit(X_train, y_train)
y_test_pred = kNeigh.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(kNeigh.score(X_test, y_test))

print('Radius Nearest Neighbors')
rNeigh = RadiusNeighborsClassifier(radius=42.0)
rNeigh.fit(X_train, y_train)
y_test_pred = rNeigh.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(rNeigh.score(X_test, y_test))

print('Decision Tree Classifier')
dTree = DecisionTreeClassifier(random_state=0)
dTree.fit(X_train, y_train)
y_test_pred = dTree.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(dTree.score(X_test, y_test))

print('Bagging (with K Nearest Neighbors)')
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train)
y_test_pred = bagging.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(bagging.score(X_test, y_test))

print('Random Forest')
rForest = RandomForestClassifier(n_estimators=100)
rForest.fit(X_train, y_train)
y_test_pred = rForest.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(rForest.score(X_test, y_test))

print('Ada Boost')
adaBoost = AdaBoostClassifier(n_estimators=10)
adaBoost.fit(X_train, y_train)
y_test_pred = adaBoost.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(adaBoost.score(X_test, y_test))

## Classification (without feature selection):
print('Without feature selection')
X = data.iloc[:,1:34]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Ridge')
lin_reg = linear_model.RidgeClassifier()
lin_reg.fit(X_train, y_train)
y_test_pred = lin_reg.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(lin_reg.score(X_test, y_test))

print('SGD')
sgdClassifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
sgdClassifier.fit(X_train,y_train)
y_test_pred = sgdClassifier.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(sgdClassifier.score(X_test, y_test))

print('Linear Discriminant Analysis svd')
linDisc = LinearDiscriminantAnalysis(solver='svd')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(linDisc.score(X_test, y_test))

print('Linear Discriminant Analysis lsqr')
linDisc = LinearDiscriminantAnalysis(solver='lsqr')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(linDisc.score(X_test, y_test))

print('Linear Discriminant Analysis eigen')
linDisc = LinearDiscriminantAnalysis(solver='eigen')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(linDisc.score(X_test, y_test))

print('Quadratic Discriminant Analysis')
quadDisc = QuadraticDiscriminantAnalysis()
quadDisc.fit(X_train, y_train)
y_test_pred = quadDisc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(quadDisc.score(X_test, y_test))

print('Kernel Ridge Regression')
kerRid = KernelRidge(alpha=1.0)
kerRid.fit(X_train, y_train)
y_test_pred = kerRid.predict(X_test)
y_test_pred = [int(round(x)) for x in y_test_pred]
print(confusion_matrix(y_test, y_test_pred))
print(kerRid.score(X_test, y_test))

print('SVC')
svc = svm.SVC()
svc.fit(X_train, y_train)
y_test_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(svc.score(X_test, y_test))

print('K Nearest Neighbors')
kNeigh = KNeighborsClassifier(n_neighbors=3)
kNeigh.fit(X_train, y_train)
y_test_pred = kNeigh.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(kNeigh.score(X_test, y_test))

print('Radius Nearest Neighbors')
rNeigh = RadiusNeighborsClassifier(radius=42.0)
rNeigh.fit(X_train, y_train)
y_test_pred = rNeigh.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(rNeigh.score(X_test, y_test))

print('Decision Tree Classifier')
dTree = DecisionTreeClassifier(random_state=0)
dTree.fit(X_train, y_train)
y_test_pred = dTree.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(dTree.score(X_test, y_test))

print('Bagging (with K Nearest Neighbors)')
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train)
y_test_pred = bagging.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(bagging.score(X_test, y_test))

print('Random Forest')
rForest = RandomForestClassifier(n_estimators=100)
rForest.fit(X_train, y_train)
y_test_pred = rForest.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(rForest.score(X_test, y_test))

print('Ada Boost')
adaBoost = AdaBoostClassifier(n_estimators=10)
adaBoost.fit(X_train, y_train)
y_test_pred = adaBoost.predict(X_test)
print(confusion_matrix(y_test, y_test_pred))
print(adaBoost.score(X_test, y_test))
