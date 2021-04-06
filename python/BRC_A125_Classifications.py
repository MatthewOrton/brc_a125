import sys
sys.path.append('/Users/morton/Documents/GitHub/icrpythonradiomics/machineLearning')

import numpy as np
import pandas as pd
from pyirr import intraclass_correlation

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
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict, permutation_test_score
from matplotlib import pyplot as plt

# read radiomics data
# dataFile = '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/BRC_A125/radiomicsDataAndTarget.csv'
dataFile = '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/BRC_A125/clinicalDataAndTarget.csv'
# dataFile = '/Users/morton/Dropbox (ICR)/CLINMAG/Radiomics/BRC_A125/clinicalRadiomicDataAndTarget.csv'

df = pd.read_csv(dataFile)

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
#X = featureSelect_correlation(threshold=0.9).fit_transform(X)


rfe = RFE(estimator=LogisticRegression())
ttestFilter = SelectKBest(f_classif)  # this just does an unpaired t-test
model = LogisticRegression(solver="liblinear", max_iter=10000)
lr = LogisticRegression(penalty='none', max_iter=10000)
#pipeline = Pipeline(steps=[('s',rfe),('m',model)])
knn = KNeighborsClassifier()
pipeline2 = Pipeline(steps=[('ss', StandardScaler()), ('f',ttestFilter), ('k',knn)])
pipeline3 = Pipeline(steps=[('ss', StandardScaler()), ('rfe', RFE(estimator=SVC(kernel="linear"))), ('svc',SVC(kernel="linear"))])

estimators = []
#estimators.append({"model": pipeline,        "name": "pipeline",          "p_grid": {"s__n_features_to_select": np.array(range(2,14))},             "scoring": "roc_auc", "result": {}})
#estimators.append({"model": pipeline2,        "name": "pipeline",          "p_grid": {"f__k": np.array(range(2,X.shape[1])), "k__n_neighbors": np.array(range(1,8))},             "scoring": "roc_auc", "result": {}})
#estimators.append({"model": pipeline3,        "name": "pipeline3",          "p_grid": {"rfe__n_features_to_select": np.array(range(1,20)), "svc__C": np.logspace(-2,3,20)},             "scoring": "roc_auc", "result": {}})
#estimators.append({"model": pipeline2,        "name": "pipeline",          "p_grid": {"f__k": [5], "k__n_neighbors": [6]},             "scoring": "roc_auc", "result": {}})
#estimators.append({"model": LogisticRegression(solver="liblinear", max_iter=10000, penalty='l2'),        "name": "Logistic",          "p_grid": {"C": np.logspace(-5,0,100)},             "scoring": "neg_log_loss", "result": {}})
estimators.append({"model": LogisticRegression(solver="liblinear", max_iter=10000),        "name": "Logistic",          "p_grid": {"C": np.logspace(-4,0,20), "penalty": ["l2"]},             "scoring": "neg_log_loss", "result": {}})
# estimators.append({"model": KNeighborsClassifier(),                                        "name": "KNN",               "p_grid": {"n_neighbors":np.array(range(1,12))},                              "scoring": "roc_auc", "result": {}})
# estimators.append({"model": GaussianNB(),                                                  "name": "GaussianNB",        "p_grid": {},                                                         "scoring": "roc_auc", "result": {}})
# estimators.append({"model": SVC(kernel="rbf"),                                             "name": "SVM-RBF",           "p_grid": {"C": np.logspace(-2,3,10), "gamma": np.logspace(-4,1,10)}, "scoring": "roc_auc", "result": {}})
# estimators.append({"model": SVC(kernel="linear"),                                          "name": "SVM-linear",        "p_grid": {"C": np.logspace(-2,3,20)},                                "scoring": "roc_auc", "result": {}})
#estimators.append({"model": RandomForestClassifier(),                                        "name": "RF",                "p_grid": {"min_samples_leaf": [4, 6, 8, 12], "max_depth":np.array(range(1,12))},                                 "scoring": "roc_auc", "result": {}})
# estimators.append({"model": XGBClassifier(use_label_encoder=False, eval_metric='logloss'), "name": "XGB",               "p_grid": {},                                                         "scoring": "roc_auc", "result": {}})
# estimators.append({"model": AdaBoostClassifier(),                                          "name": "AdaBoost",          "p_grid": {},                                                         "scoring": "roc_auc", "result": {}})
# estimators.append({"model": PassiveAggressiveClassifier(),                                 "name": "PassiveAggressive", "p_grid": {},                                                         "scoring": "roc_auc", "result": {}})
# estimators.append({"model": QuadraticDiscriminantAnalysis(),                               "name": "QDA",               "p_grid": {},                                                         "scoring": "roc_auc", "result": {}})
# estimators.append({"model": LinearDiscriminantAnalysis(),                                  "name": "LDA",               "p_grid": {},                                                         "scoring": "roc_auc", "result": {}})
# estimators.append({"model": GaussianProcessClassifier(),                                   "name": "GaussianProcess",   "p_grid": {"kernel": [RBF(l) for l in np.logspace(-1, 1, 10)]},     "scoring": "roc_auc", "result": {}})

# estimators = nestedCVclassification(X, y, estimators, useStratified=True, n_splits_outer = 5, n_splits_inner = 5, n_repeats=20, verbose=1, linewidth=3, staircase=False, color='green', plot_individuals=True)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=12345)

clf = GridSearchCV(estimator=LogisticRegression(solver="liblinear", max_iter=10000, penalty='l2'), param_grid={"C": np.logspace(-4,0,20)}, cv=inner_cv, refit=True, verbose=0, scoring="neg_log_loss")

# cv_result = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring='roc_auc', return_estimator=True, verbose=1, n_jobs=-1)
score, perm_scores, pvalue = permutation_test_score(clf, X, y, scoring="roc_auc", cv=outer_cv, n_permutations=10000 , verbose=4, n_jobs=-1)

import time
time.sleep(1)

print(' ')
print('_______________________________________')
print('AUCROC      = ' + str(round(score,3)))
print('AUCROC perm = ' + str(np.mean(perm_scores).round(3)) + ' \u00B1 ' + str(np.std(perm_scores).round(3)))
print('p-value     = ' + str(pvalue.round(5)))

plt.hist(perm_scores)
plt.show()