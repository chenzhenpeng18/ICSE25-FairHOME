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
import copy
from utility import get_classifier
from aif360.datasets import BinaryLabelDataset
import tensorflow as tf

def Linear_regression(x, slope, intercept):
    return x * slope + intercept

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['compas_new'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['dl'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'compas_new': ['sex','race','age']}

val_name = "fairhome5_{}_{}.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()

sa0 = macro_var[dataset_used][0]
sa1 = macro_var[dataset_used][1]
sa2 = macro_var[dataset_used][2]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'spd2', 'aod2', 'eod2', 'wcspd', 'wcaod', 'wceod', 'avespd', 'aveaod', 'aveeod']
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

    # X_test2 = X_test.copy()
    # X_test3 = X_test.copy()
    # X_test4 = X_test.copy()
    X_test5 = X_test.copy()
    X_test6 = X_test.copy()
    X_test7 = X_test.copy()
    X_test8 = X_test.copy()

    for rown in range(len(X_test)):
        # X_test2.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        #
        # X_test3.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]
        #
        # X_test4.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]

        X_test5.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        X_test5.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]

        X_test6.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        X_test6.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]

        X_test7.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]
        X_test7.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]

        X_test8.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        X_test8.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]
        X_test8.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]

    clf = get_classifier(clf_name,(X_train.shape[1],))
    clf.fit(X_train, y_train,epochs=20)
    pred_de1 = clf.predict_classes(X_test)
    # pred_de2 = clf.predict_classes(X_test2)
    # pred_de3 = clf.predict_classes(X_test3)
    # pred_de4 = clf.predict_classes(X_test4)
    pred_de5 = clf.predict_classes(X_test5)
    pred_de6 = clf.predict_classes(X_test6)
    pred_de7 = clf.predict_classes(X_test7)
    pred_de8 = clf.predict_classes(X_test8)

    res = []
    for i in range(len(pred_de1)):
        count1 = pred_de1[i] + pred_de5[i] + pred_de6[i] + pred_de7[i] + pred_de8[i]
        count0 = 5 - count1
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