import sys
import os
sys.path.append(os.path.abspath('.'))
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
from Measure_new import measure_final_score
from aif360.datasets import BinaryLabelDataset
import time
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['compas_new'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['lr'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'compas_new': ['sex','race','age']}

val_name = "adv_{}_{}.txt".format(clf_name,dataset_used)
fout_time = open("adv_{}_{}_time.txt".format(clf_name, dataset_used), 'w')
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
privileged_groups = [{macro_var[dataset_used][0]: 1}, {macro_var[dataset_used][1]: 1}, {macro_var[dataset_used][2]: 1}]
unprivileged_groups = [{macro_var[dataset_used][0]: 0}, {macro_var[dataset_used][1]: 0}, {macro_var[dataset_used][2]: 0}]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'spd2', 'aod2', 'eod2', 'wcspd', 'wcaod', 'wceod', 'avespd', 'aveaod', 'aveeod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 20
for r in range(repeat_time):
    print(r)
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
    tf.reset_default_graph()
    sess = tf.Session()
    scope = "clf"+str(r)
    adversarial  = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name=scope,
                          debias=True,
                          sess=sess)
    adversarial = adversarial.fit(dataset_orig_train)
    pred_ad = adversarial.predict(dataset_orig_test)

    round_result = measure_final_score(dataset_orig_test, pred_ad, macro_var[dataset_used])
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