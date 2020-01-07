#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.impute import SimpleImputer
import missingno as msno

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
#msno.matrix(data).figure.savefig('output.png')



X = data.iloc[:,1:34]
y = data.iloc[:,0]

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(10, 'Score'))
