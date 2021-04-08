import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics/machineLearning')

import numpy as np
import pandas as pd
from pyirr import intraclass_correlation
import os

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from featureSelect_correlation import featureSelect_correlation

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from nestedCVclassification import nestedCVclassification
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict, permutation_test_score, permutation_test_type2_score
from matplotlib import pyplot as plt

# read radiomics data
dataFolder = '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/BRC_A125'
dataFiles = []
# dataFiles.append('radiomics')
# dataFiles.append('clinical_radiomics')
dataFiles.append('clinical')


logReg = LogisticRegression(solver="liblinear", max_iter=10000, penalty='l1')

estimators = []
estimators.append({"model": logReg, "name": "Logistic+ridge", "p_grid": {"C": np.logspace(-4, 0, 20)}, "scoring": "neg_log_loss",     "result": {}})
# estimators.append({"model": GaussianNB(), "name": "GaussianNB", "p_grid": {}, "scoring": "roc_auc", "result": {}})
# estimators.append({"model": KNeighborsClassifier(), "name": "KNN", "p_grid": {"n_neighbors": np.array(range(1, 12))},                   "scoring": "roc_auc", "result": {}})
# estimators.append({"model": SVC(kernel="rbf"), "name": "SVM-RBF",                   "p_grid": {"C": np.logspace(-2, 3, 8), "gamma": np.logspace(-4, 1, 8)}, "scoring": "roc_auc",                   "result": {}})

# estimators.append({"model": RandomForestClassifier(), "name": "RF",             "p_grid": {"min_samples_leaf": [4, 6, 8, 10], "max_depth":[2, 4, 6, 8, 10]}, "scoring": "roc_auc",      "result": {}})
# estimators.append({"model": SVC(kernel="linear"),     "name": "SVM-linear",     "p_grid": {"C": np.logspace(-2,3,20)},                                "scoring": "roc_auc", "result": {}})

for estimator in estimators:

    for dataFile in dataFiles:

        print(' ')
        print('_______________________________________')
        print(dataFile + ' // ' + estimator["name"])

        df = pd.read_csv(os.path.join(dataFolder, dataFile+'.csv'))


        targetStr = "simplifiedNodalStatus"

        X = np.array(df.loc[:,~df.columns.str.match(targetStr)])
        y = np.array(df[targetStr])

        # log transform any features that are all positive or all negative
        # for n, column in enumerate(X.T):
        #     if np.all(np.sign(column)==np.sign(column[0])):
        #         X[:,n] = np.log(np.abs(column))


        # pre-process data
        X = VarianceThreshold().fit_transform(X)
        X = StandardScaler().fit_transform(X)
        # X = featureSelect_correlation(threshold=0.9).fit_transform(X)


        # estimators = nestedCVclassification(X, y, estimators, useStratified=True, n_splits_outer = 5, n_splits_inner = 5, n_repeats=20, verbose=1, linewidth=3, staircase=False, color='green', plot_individuals=True)
        # cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring='roc_auc', return_estimator=True, verbose=1, n_jobs=-1)
        # score, perm_scores, pvalue = permutation_test_type2_score(clf, X, y, scoring="roc_auc", cv=outer_cv, n_permutations=200 , verbose=4, n_jobs=-1)

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=12345)

        n_permutations = 1000

        clf = GridSearchCV(estimator=estimator["model"], param_grid=estimator["p_grid"], cv=inner_cv, refit=True, verbose=0, scoring=estimator["scoring"])
        score, perm_scores, pvalue = permutation_test_score(clf, X, y, scoring="roc_auc", cv=outer_cv, n_permutations=n_permutations, verbose=7, n_jobs=-1)

        print(' ')
        print('_______________________________________')
        print(dataFile + ' // ' + estimator["name"])
        print('AUCROC      = ' + str(round(score,4)))
        print('AUCROC perm = ' + str(np.mean(perm_scores).round(4)) + ' \u00B1 ' + str(np.std(perm_scores).round(4)))
        print('p-value     = ' + str(pvalue.round(5)) + ' = (' + str(int(pvalue*(n_permutations+1)-1)) + ' + 1)/(' + str(n_permutations) + ' + 1)')

