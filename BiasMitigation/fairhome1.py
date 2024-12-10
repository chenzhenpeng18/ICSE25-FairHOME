import sys
import os
sys.path.append(os.path.abspath('.'))
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Measure_new import measure_final_score
import argparse
from aif360.datasets import BinaryLabelDataset
import copy
from utility import get_classifier
from sklearn.linear_model import LinearRegression

def Linear_regression(x, slope, intercept):
    return x * slope + intercept


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'default', 'mep1', 'mep2','german'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'adult': ['sex','race'],  'default':['sex','age'], 'mep1': ['sex','race'],'mep2': ['sex','race'],'german': ['sex','age']}

val_name = "fairhome1_{}_{}.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()

sa0 = macro_var[dataset_used][0]
sa1 = macro_var[dataset_used][1]

privileged_groups = [{macro_var[dataset_used][0]: 1}, {macro_var[dataset_used][1]: 1}]
unprivileged_groups = [{macro_var[dataset_used][0]: 0}, {macro_var[dataset_used][1]: 0}]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'wcspd', 'wcaod', 'wceod', 'avespd', 'aveaod', 'aveeod']
for p_index in performance_index:
    results[p_index] = []


repeat_time = 20
for r in range(repeat_time):
    print (r)
    np.random.seed(r)

    # split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
    y_train = copy.deepcopy(dataset_orig_train['Probability'])
    X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
    y_test = copy.deepcopy(dataset_orig_test['Probability'])

    X_test2 = X_test.copy()
    X_test3 = X_test.copy()
    X_test4 = X_test.copy()

    for rown in range(len(X_test)):
        X_test2.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]

        X_test3.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]

        X_test4.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        X_test4.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]

    column_train = [column for column in X_train]
    for i in column_train:
        if i != sa0 and i != sa1:
            tmp_reg = LinearRegression().fit(X_train[[sa0, sa1]], X_train[i])
            X_test2[i] = X_test[i] - tmp_reg.predict(X_test[[sa0, sa1]]) + tmp_reg.predict(X_test2[[sa0, sa1]])
            X_test3[i] = X_test[i] - tmp_reg.predict(X_test[[sa0, sa1]]) + tmp_reg.predict(X_test3[[sa0, sa1]])
            X_test4[i] = X_test[i] - tmp_reg.predict(X_test[[sa0, sa1]]) + tmp_reg.predict(X_test4[[sa0, sa1]])

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)

    pred_de1 = clf.predict(X_test)
    pred_de2 = clf.predict(X_test2)
    pred_de3 = clf.predict(X_test3)
    pred_de4 = clf.predict(X_test4)

    res = []
    for i in range(len(pred_de1)):
        count1 = pred_de1[i] + pred_de2[i] + pred_de3[i] + pred_de4[i]
        count0 = 4 - count1
        if count1 >= count0:
            res.append(1)
        else:
            res.append(0)

    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=macro_var[dataset_used])
    test_df_copy = copy.deepcopy(dataset_orig_test)
    test_df_copy.labels = np.array(res).reshape(-1, 1)
    round_result = measure_final_score(dataset_orig_test, test_df_copy, macro_var[dataset_used])

    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()