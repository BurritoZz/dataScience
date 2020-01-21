#usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from sklearn.kernel_ridge import KernelRidge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
# import seaborn as sns
# import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


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


def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
    :param series: Pandas DataFrame object
    :Return: float
    """
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 2)

#print(get_percentage_missing(data))

# Data preparation:
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
#print(top_corr_features)
#plt.figure(figsize=(40,40))
#g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.savefig('hetejongen')


## Classification (with feature selection):
print('With feature selection')
X = data[['Age', 'leg_left_pain_intensity', 'leg_right_pain_intensity', 'arm_right_pain_intensity', 'Decreased_mobility', 'neck_pain_intensity', 'Irrational_thoughts_work', 'Fever', 'Irrational_thoughts_risk_lasting', 'Coping_strategy', 'Relationship_with_colleagues', 'low_back_pain_intensity', 'Extremely_nervous', 'Kinesiophobia_pain_stop', 'Weightloss_per_year']]
y = data['Treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_selection_performance = []

print('Ridge')
lin_reg = linear_model.RidgeClassifier()
lin_reg.fit(X_train, y_train)
y_test_pred = lin_reg.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = lin_reg.score(X_test, y_test)
feature_selection_performance.append(('Ridge', score, matrix))

#print('SGD')
#sgdClassifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#sgdClassifier.fit(X_train,y_train)
#y_test_pred = sgdClassifier.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = sgdClassifier.score(X_test, y_test)
#feature_selection_performance.append(('SGD', score, matrix))

#print('Linear Discriminant Analysis svd')
#linDisc = LinearDiscriminantAnalysis(solver='svd')
#linDisc.fit(X_train, y_train)
#y_test_pred = linDisc.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = linDisc.score(X_test, y_test)
#feature_selection_performance.append(('Linear Discriminant Analysis svd', score, matrix))

print('Linear Discriminant Analysis lsqr')
linDisc = LinearDiscriminantAnalysis(solver='lsqr')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = linDisc.score(X_test, y_test)
feature_selection_performance.append(('Linear Discriminant Analysis lsqr', score, matrix))

#print('Linear Discriminant Analysis eigen')
#linDisc = LinearDiscriminantAnalysis(solver='eigen')
#linDisc.fit(X_train, y_train)
#y_test_pred = linDisc.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = linDisc.score(X_test, y_test)
#feature_selection_performance.append(('Linear Discriminant Analysis eigen', score, matrix))

#print('Quadratic Discriminant Analysis')
#quadDisc = QuadraticDiscriminantAnalysis()
#quadDisc.fit(X_train, y_train)
#y_test_pred = quadDisc.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = quadDisc.score(X_test, y_test)
#feature_selection_performance.append(('Quadratic Discriminant Analysis', score, matrix))

#print('Kernel Ridge Regression')
#kerRid = KernelRidge(alpha=1.0)
#kerRid.fit(X_train, y_train)
#y_test_pred = kerRid.predict(X_test)
#y_test_pred = [int(round(x)) for x in y_test_pred]
#matrix = confusion_matrix(y_test, y_test_pred)
#score = kerRid.score(X_test, y_test)
#feature_selection_performance.append(('Kernel Ridge Regression', score, matrix))

print('SVC')
svc = svm.SVC(gamma='scale')
svc.fit(X_train, y_train)
y_test_pred = svc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = svc.score(X_test, y_test)
feature_selection_performance.append(('SVC', score, matrix))

"""
#=======================================================================
#print(svc.get_params())
parameters = {'C':[1, 10, 100, 1000],
              'gamma':['scale', 'auto']}

clf = GridSearchCV(svm.SVC(), parameters)
clf.fit(X_train, y_train)

print("Best scores found for parameters set:")
print("%0.3f for %r" % (clf.best_score_, clf.best_params_))
print()

print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f)) for %r" % (mean, std * 2, params))
print()
#=======================================================================
"""

#print('LinearSVC')
#lin_svc = svm.LinearSVC(max_iter=10000)
#lin_svc.fit(X_train,y_train)
#y_test_pred = lin_svc.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = lin_svc.score(X_test, y_test)
#feature_selection_performance.append(('LinearSVC', score, matrix))

#print("Multinomial Naive Bayes")
#multNB = naive_bayes.MultinomialNB()
#multNB.fit(X_train,y_train)
#y_test_pred = multNB.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = multNB.score(X_test,y_test)
#feature_selection_performance.append(('Multinomial Naive Bayes', score, matrix))

#print('Complement Naive Bayes')
#compNB = naive_bayes.ComplementNB()
#compNB.fit(X_train, y_train)
#y_test_pred = compNB.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = compNB.score(X_test, y_test)
#feature_selection_performance.append(('Complement Naive Bayes', score, matrix))

#print('Gradient Boosting Classifier')
#gradBoost = ensemble.GradientBoostingClassifier()
#gradBoost.fit(X_train, y_train)
#y_test_pred = gradBoost.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = gradBoost.score(X_test, y_test)
#feature_selection_performance.append(('Gradient Boosting Classifier', score, matrix))

#print('K Nearest Neighbors')
#kNeigh = KNeighborsClassifier(n_neighbors=3)
#kNeigh.fit(X_train, y_train)
#y_test_pred = kNeigh.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = kNeigh.score(X_test, y_test)
#feature_selection_performance.append(('K Nearest Neighbours', score, matrix))

"""
#=======================================================================
parameters = {'n_neighbors':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'weights':['uniform', 'distance'],
              'algorithm':['ball_tree', 'kd_tree', 'brute']}

clf = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

print("Best scores found for parameters set:")
print("%0.3f for %r" % (clf.best_score_, clf.best_params_))
print()

print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f)) for %r" % (mean, std * 2, params))
print()
#=======================================================================
"""

#print('Radius Nearest Neighbors')
#rNeigh = RadiusNeighborsClassifier(radius=42.0)
#rNeigh.fit(X_train, y_train)
#y_test_pred = rNeigh.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = rNeigh.score(X_test, y_test)
#feature_selection_performance.append(('Radius Nearest Neighbours', score, matrix))

#print('Decision Tree Classifier')
#dTree = DecisionTreeClassifier(random_state=0)
#dTree.fit(X_train, y_train)
#y_test_pred = dTree.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = dTree.score(X_test, y_test)
#feature_selection_performance.append(('Decision Tree Classifier', score, matrix))

print('Bagging (with K Nearest Neighbors)')
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train)
y_test_pred = bagging.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = bagging.score(X_test, y_test)
feature_selection_performance.append(('Bagging K Nearest Neigbours', score, matrix))

"""
#=======================================================================
parameters = {'base_estimator':[KNeighborsClassifier()],
              'max_samples':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              'max_features':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

clf = GridSearchCV(BaggingClassifier(), parameters)
clf.fit(X, y)

print("Best scores found for parameters set:")
print("%0.3f for %r" % (clf.best_score_, clf.best_params_))
print()

print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f)) for %r" % (mean, std * 2, params))
print()
#=======================================================================
"""
print('Bagging (with SVC')
bagging2 = BaggingClassifier(svm.SVC(), max_samples=0.5, max_features=0.5)
bagging2.fit(X_train, y_train)
y_test_pred = bagging2.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = bagging2.score(X_test, y_test)
print('Bagging with SVC:' + str(score))

print('Random Forest')
rForest = ensemble.RandomForestClassifier(n_estimators=100)
rForest.fit(X_train, y_train)
y_test_pred = rForest.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = rForest.score(X_test, y_test)
feature_selection_performance.append(('Random Forest', score, matrix))


#=======================================================================
parameters = {'n_estimators':[1, 10, 100, 1000],
              'criterion':['gini', 'entropy']}

clf = GridSearchCV(ensemble.RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

print("Best scores found for parameters set:")
print("%0.3f for %r" % (clf.best_score_, clf.best_params_))
print()

print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f)) for %r" % (mean, std * 2, params))
print()
#=======================================================================


#print('Ada Boost')
#adaBoost = AdaBoostClassifier(n_estimators=10)
#adaBoost.fit(X_train, y_train)
#y_test_pred = adaBoost.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = adaBoost.score(X_test, y_test)
#feature_selection_performance.append(('Ada Boost', score, matrix))

# Classification (without feature selection):
print('Without feature selection')
X_train = data.iloc[:,1:34]
y_train = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
no_selection_performance = []

print('Ridge')
lin_reg = linear_model.RidgeClassifier()
lin_reg.fit(X_train, y_train)
y_test_pred = lin_reg.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = lin_reg.score(X_test, y_test)
no_selection_performance.append(('Ridge', score, matrix))

#print('SGD')
#sgdClassifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
#sgdClassifier.fit(X_train,y_train)
#y_test_pred = sgdClassifier.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = sgdClassifier.score(X_test, y_test)
#no_selection_performance.append(('SGD', score, matrix))

#print('Linear Discriminant Analysis svd')
#linDisc = LinearDiscriminantAnalysis(solver='svd')
#linDisc.fit(X_train, y_train)
#y_test_pred = linDisc.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = linDisc.score(X_test, y_test)
#no_selection_performance.append(('Linear Discrimitnant Analysis svd', score, matrix))

print('Linear Discriminant Analysis lsqr')
linDisc = LinearDiscriminantAnalysis(solver='lsqr')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = linDisc.score(X_test, y_test)
no_selection_performance.append(('Linear Discriminant Analysis lsqr', score, matrix))

#print('Linear Discriminant Analysis eigen')
#linDisc = LinearDiscriminantAnalysis(solver='eigen')
#linDisc.fit(X_train, y_train)
#y_test_pred = linDisc.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = linDisc.score(X_test, y_test)
#no_selection_performance.append(('Linear Discriminant Analysis eigen', score, matrix))

#print('Quadratic Discriminant Analysis')
#quadDisc = QuadraticDiscriminantAnalysis()
#quadDisc.fit(X_train, y_train)
#y_test_pred = quadDisc.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = quadDisc.score(X_test, y_test)
#no_selection_performance.append(('Quadratic Discriminant Analysis', score, matrix))

#print('Kernel Ridge Regression')
#kerRid = KernelRidge(alpha=1.0)
#kerRid.fit(X_train, y_train)
#y_test_pred = kerRid.predict(X_test)
#y_test_pred = [int(round(x)) for x in y_test_pred]
#matrix = confusion_matrix(y_test, y_test_pred)
#score = kerRid.score(X_test, y_test)
#no_selection_performance.append(('Kernel Ridge Regression', score, matrix))

print('SVC')
svc = svm.SVC(gamma='scale')
svc.fit(X_train, y_train)
y_test_pred = svc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = svc.score(X_test, y_test)
no_selection_performance.append(('SVC', score, matrix))

print('LinearSVC')
lin_svc = svm.LinearSVC(max_iter=10000)
lin_svc.fit(X_train, y_train)
y_test_pred = lin_svc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = lin_svc.score(X_test, y_test)
no_selection_performance.append(('LinearSVC', score, matrix))

#print("Multinomial Naive Bayes")
#multNB = naive_bayes.MultinomialNB()
#multNB.fit(X_train,y_train)
#y_test_pred = multNB.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = multNB.score(X_test,y_test)
#no_selection_performance.append(('Multinomial Naive Bayes', score, matrix))

#print('Complement Naive Bayes')
#compNB = naive_bayes.ComplementNB()
#compNB.fit(X_train, y_train)
#y_test_pred = compNB.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = compNB.score(X_test, y_test)
#no_selection_performance.append(('Complement Naive Bayes', score, matrix))

#print('Gradient Boosting Classifier')
#gradBoost = ensemble.GradientBoostingClassifier()
#gradBoost.fit(X_train, y_train)
#y_test_pred = gradBoost.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = gradBoost.score(X_test, y_test)
#no_selection_performance.append(('Gradient Boosting Classifier', score, matrix))

#print('K Nearest Neighbors')
#kNeigh = KNeighborsClassifier(n_neighbors=3)
#kNeigh.fit(X_train, y_train)
#y_test_pred = kNeigh.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = kNeigh.score(X_test, y_test)
#no_selection_performance.append(('K Nearest Neighbours', score, matrix))

#print('Radius Nearest Neighbors')
#rNeigh = RadiusNeighborsClassifier(radius=42.0)
#rNeigh.fit(X_train, y_train)
#y_test_pred = rNeigh.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = rNeigh.score(X_test, y_test)
#no_selection_performance.append(('Radius Nearest Neighbours', score, matrix))

#print('Decision Tree Classifier')
#dTree = DecisionTreeClassifier(random_state=0)
#dTree.fit(X_train, y_train)
#y_test_pred = dTree.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = dTree.score(X_test, y_test)
#no_selection_performance.append(('Decision Tree Classifier', score, matrix))

#print('Bagging (with K Nearest Neighbors)')
#bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
#bagging.fit(X_train, y_train)
#y_test_pred = bagging.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = bagging.score(X_test, y_test)
#no_selection_performance.append(('Bagging with K Nearest Neighbours', score, matrix))


n_estimator = [10, 100, 1000]
criterion = ["gini", "entropy"]
max_depth = ["None", 10, 100]
min_samples_split = [2, 4, 8]

print('Random Forest')
rForest = ensemble.RandomForestClassifier(n_estimators=100)
rForest.fit(X_train, y_train)
y_test_pred = rForest.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = rForest.score(X_test, y_test)
no_selection_performance.append(('Random Forest', score, matrix))

#print('Ada Boost')
#adaBoost = AdaBoostClassifier(n_estimators=10)
#adaBoost.fit(X_train, y_train)
#y_test_pred = adaBoost.predict(X_test)
#matrix = confusion_matrix(y_test, y_test_pred)
#score = adaBoost.score(X_test, y_test)
#no_selection_performance.append(('Ada Boost', score, matrix))

feature_selection_performance = sorted(feature_selection_performance, key=lambda x: x[1])
no_selection_performance = sorted(no_selection_performance, key=lambda x: x[1])

print()

print("With feature selection:")
for e in feature_selection_performance:
    print(e[0] + ": " + str(e[1]))

print()

print("Without feature selection:")
for e in no_selection_performance:
    print(e[0] + ": " + str(e[1]))

#import autosklearn.classification
#import sklearn.model_selection
#import sklearn.datasets
#import sklearn.metrics
#automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=86400, per_run_time_limit=1800, seed=420, ml_memory_limit=8192, )
#automl.fit(X_train, y_train)
#y_hat = automl.predict(X_test)
#print('Accuracy score', sklearn.metrics.accuracy_score(y_test, y_hat))
#
#from joblib import dump, load
#dump(automl, 'best_classifier.joblib')
