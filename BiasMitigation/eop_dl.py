import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_classifier
from sklearn.model_selection import train_test_split
import argparse
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
import tensorflow as tf
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'default', 'mep1', 'mep2','german'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['dl'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'adult': ['sex','race'], 'default':['sex','age'], 'mep1': ['sex','race'],'mep2': ['sex','race'],'german': ['sex','age']}

val_name = "eop_{}_{}.txt".format(clf_name,dataset_used)
fout_time = open("eop_{}_{}_time.txt".format(clf_name, dataset_used), 'w')
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
privileged_groups = [{macro_var[dataset_used][0]: 1}, {macro_var[dataset_used][1]: 1}]
unprivileged_groups = [{macro_var[dataset_used][0]: 0}, {macro_var[dataset_used][1]: 0}]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'wcspd', 'wcaod', 'wceod', 'avespd', 'aveaod', 'aveeod']
for p_index in performance_index:
    results[p_index] = []

randseed = 12345679
repeat_time = 20
for r in range(repeat_time):
    print (r)
    np.random.seed(r)

    # split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                            label_names=['Probability'],
                                            protected_attribute_names=macro_var[dataset_used])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=macro_var[dataset_used])

    start_time = time.time()
    clf = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
    clf.fit(dataset_orig_train.features, dataset_orig_train.labels, epochs=20)

    train_pred = clf.predict_classes(dataset_orig_train.features).reshape(-1, 1)
    train_prob = clf.predict(dataset_orig_train.features).reshape(-1, 1)

    pred = clf.predict_classes(dataset_orig_test.features).reshape(-1, 1)
    pred_prob = clf.predict(dataset_orig_test.features).reshape(-1, 1)

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = train_pred
    dataset_orig_train_pred.scores = train_prob

    dataset_orig_test_pred = dataset_orig_test.copy()
    dataset_orig_test_pred.labels = pred
    dataset_orig_test_pred.scores = pred_prob
    
    eqo = EqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     seed=randseed)
    eqo = eqo.fit(dataset_orig_train, dataset_orig_train_pred)
    pred_eqo = eqo.predict(dataset_orig_test_pred)

    round_result = measure_final_score(dataset_orig_test, pred_eqo, macro_var[dataset_used])
    end_time = time.time()
    fout_time.write('%f\n' % (end_time - start_time))
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()
fout_time.close()