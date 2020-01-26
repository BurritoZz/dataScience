#usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
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
import pickle
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
data = data.drop(columns=['Relationship_with_colleagues'])
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
data = data.drop(columns=['Trauma'])
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
#
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


#corrmat = data.corr()
#top_corr_features = corrmat.index
#print(top_corr_features)
#plt.figure(figsize=(40,40))
#g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
#plt.savefig('hetejongen')

#def decimalToBinary(n):
#    b=0
#    i=1
#    while(n != 0):
#        r=n%2
#        b+=r*i
#        n//=2
#        i=i*10
#    return b
#
#def makeList(k):
#    a=[]
#    if(k==0):
#        a.append(0)
#
#    while(k>0):
#        a.append(k%10)
#        k//=10
#    a.reverse()
#    return a
#
#def checkBinary(bin, l):
#    temp=[]
#    for i in range(len(bin)):
#        if(bin[i]==1):
#            temp.append(l[i])
#    return temp
#
## Classification (with feature selection):
print('With feature selection')
selected_features = ['leg_right_pain_intensity', 'leg_left_pain_intensity', 'arg_right_pain_intensity', 'neck_pain_intensity', 'Kinesiophobia_physical_exercise', 'Kinesiophabia_pain_stop', 'Irrational_thoughts_work', 'Extremely_nervous', 'Age', 'Irrational_thoughts_risk_lasting', 'low_back_pain_intensity', 'Coping_strategy', 'Decreased_mobility', 'Fever', 'Serious_disease', 'Paidwork']
#
#binlist=[]
#subsets=[]
#n=len(selected_features)
#
#for i in range(2**n):
#    s=decimalToBinary(i)
#    arr=makeList(s)
#    binlist.append(arr)
#
#    for i in binlist:
#        k=0
#        while(len(i)!=n):
#            i.insert(k,0)
#            k = k+1
#for i in binlist:
#    subsets.append(checkBinary(i,selected_features))
#
#with open('subsetlist', 'wb') as fp:
#    pickle.dump(subsets, fp)

with open('subsetlist', 'rb') as fp:
    featureSubsets = pickle.load(fp)

print(len(featureSubsets))

X = data[featurelist]
y = data['Treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_selection_performance = []


# Classification (without feature selection):
print('Without feature selection')
X = data.iloc[:,1:34]
y = data.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
no_selection_performance = []

print('Ridge')
lin_reg = linear_model.RidgeClassifier(alpha=1000, fit_intercept=True, normalize=False, solver='lsqr', tol=1e-2)
lin_reg.fit(X_train, y_train)
y_test_pred = lin_reg.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = lin_reg.score(X_test, y_test)
no_selection_performance.append(('Ridge', score, matrix))

print('SGD')
sgdClassifier = linear_model.SGDClassifier(fit_intercept=True, loss='log', max_iter=1000, penalty='l1', shuffle=False, tol=0.01)
sgdClassifier.fit(X_train,y_train)
y_test_pred = sgdClassifier.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = sgdClassifier.score(X_test, y_test)
no_selection_performance.append(('SGD', score, matrix))

print('Linear Discriminant Analysis svd')
linDisc = LinearDiscriminantAnalysis(solver='svd')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = linDisc.score(X_test, y_test)
no_selection_performance.append(('Linear Discrimitnant Analysis svd', score, matrix))

print('Linear Discriminant Analysis lsqr')
linDisc = LinearDiscriminantAnalysis(solver='eigen', store_covariance=True, tol=1e-2)
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = linDisc.score(X_test, y_test)
no_selection_performance.append(('Linear Discriminant Analysis lsqr', score, matrix))

print('Linear Discriminant Analysis eigen')
linDisc = LinearDiscriminantAnalysis(solver='eigen')
linDisc.fit(X_train, y_train)
y_test_pred = linDisc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = linDisc.score(X_test, y_test)
no_selection_performance.append(('Linear Discriminant Analysis eigen', score, matrix))

print('Quadratic Discriminant Analysis')
quadDisc = QuadraticDiscriminantAnalysis()
quadDisc.fit(X_train, y_train)
y_test_pred = quadDisc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = quadDisc.score(X_test, y_test)
no_selection_performance.append(('Quadratic Discriminant Analysis', score, matrix))

print('Kernel Ridge Regression')
kerRid = KernelRidge(alpha=1.0)
kerRid.fit(X_train, y_train)
y_test_pred = kerRid.predict(X_test)
y_test_pred = [int(round(x)) for x in y_test_pred]
matrix = confusion_matrix(y_test, y_test_pred)
score = kerRid.score(X_test, y_test)
no_selection_performance.append(('Kernel Ridge Regression', score, matrix))

print('SVC')
svc = svm.SVC(C=1, class_weight=None, coef0=0, gamma='scale', kernel='rbf', shrinking=True, tol=1e-1)
svc.fit(X_train, y_train)
y_test_pred = svc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = svc.score(X_test, y_test)
no_selection_performance.append(('SVC', score, matrix))

print('LinearSVC')
lin_svc = svm.LinearSVC(C=5, class_weight='balanced', loss='squared_hinge', tol=0.0001, max_iter=100000)
lin_svc.fit(X_train, y_train)
y_test_pred = lin_svc.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = lin_svc.score(X_test, y_test)
no_selection_performance.append(('LinearSVC', score, matrix))

print("Multinomial Naive Bayes")
multNB = naive_bayes.MultinomialNB()
multNB.fit(X_train,y_train)
y_test_pred = multNB.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = multNB.score(X_test,y_test)
no_selection_performance.append(('Multinomial Naive Bayes', score, matrix))

print('Complement Naive Bayes')
compNB = naive_bayes.ComplementNB()
compNB.fit(X_train, y_train)
y_test_pred = compNB.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = compNB.score(X_test, y_test)
no_selection_performance.append(('Complement Naive Bayes', score, matrix))

print('Gradient Boosting Classifier')
gradBoost = ensemble.GradientBoostingClassifier()
gradBoost.fit(X_train, y_train)
y_test_pred = gradBoost.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = gradBoost.score(X_test, y_test)
no_selection_performance.append(('Gradient Boosting Classifier', score, matrix))

print('K Nearest Neighbors')
kNeigh = KNeighborsClassifier(n_neighbors=3)
kNeigh.fit(X_train, y_train)
y_test_pred = kNeigh.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = kNeigh.score(X_test, y_test)
no_selection_performance.append(('K Nearest Neighbours', score, matrix))

print('Radius Nearest Neighbors')
rNeigh = RadiusNeighborsClassifier(radius=42.0)
rNeigh.fit(X_train, y_train)
y_test_pred = rNeigh.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = rNeigh.score(X_test, y_test)
no_selection_performance.append(('Radius Nearest Neighbours', score, matrix))

print('Decision Tree Classifier')
dTree = DecisionTreeClassifier(random_state=0)
dTree.fit(X_train, y_train)
y_test_pred = dTree.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = dTree.score(X_test, y_test)
no_selection_performance.append(('Decision Tree Classifier', score, matrix))

print('Bagging (with K Nearest Neighbors)')
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train, y_train)
y_test_pred = bagging.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = bagging.score(X_test, y_test)
no_selection_performance.append(('Bagging with K Nearest Neighbours', score, matrix))

print('Random Forest')
rForest = ensemble.RandomForestClassifier(n_estimators=1000, criterion='gini', max_features='log2', min_samples_leaf=3, min_samples_split=3)
rForest.fit(X_train, y_train)
y_test_pred = rForest.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = rForest.score(X_test, y_test)
no_selection_performance.append(('Random Forest', score, matrix))

print('Ada Boost')
adaBoost = AdaBoostClassifier(n_estimators=10)
adaBoost.fit(X_train, y_train)
y_test_pred = adaBoost.predict(X_test)
matrix = confusion_matrix(y_test, y_test_pred)
score = adaBoost.score(X_test, y_test)
no_selection_performance.append(('Ada Boost', score, matrix))

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
