import sys
import os
sys.path.append(os.path.abspath('.'))
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import copy
from WAE import data_dis
from aif360.datasets import BinaryLabelDataset
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from utility import get_classifier
from Measure_new import measure_final_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['compas_new'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['rf', 'svm', 'lr'], help="Classifier name")

    args = parser.parse_args()
    dataset_used = args.dataset
    clf_name = args.clf

    macro_var = {'compas_new': ['sex','race','age']}

    val_name = "maat_{}_{}.txt".format(clf_name,dataset_used)
    fout = open(val_name, 'w')

    dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()
    privileged_groups = [{macro_var[dataset_used][0]: 1}, {macro_var[dataset_used][1]: 1},
                         {macro_var[dataset_used][2]: 1}]
    unprivileged_groups = [{macro_var[dataset_used][0]: 0}, {macro_var[dataset_used][1]: 0},
                           {macro_var[dataset_used][2]: 0}]

    results = {}
    performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'spd2', 'aod2', 'eod2', 'wcspd', 'wcaod', 'wceod', 'avespd', 'aveaod', 'aveeod']
    for p_index in performance_index:
        results[p_index] = []

    repeat_time = 20

    for r in range(repeat_time):
        print (r)

        np.random.seed(r)
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        dataset_orig_train_new_for_attr1 = data_dis(pd.DataFrame(dataset_orig_train),macro_var[dataset_used][0])
        dataset_orig_train_new_for_attr2 = data_dis(pd.DataFrame(dataset_orig_train), macro_var[dataset_used][1])
        dataset_orig_train_new_for_attr3 = data_dis(pd.DataFrame(dataset_orig_train), macro_var[dataset_used][2])

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr1)
        dataset_orig_train_new_for_attr1 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr1), columns=dataset_orig.columns)
        dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr2)
        dataset_orig_train_new_for_attr2 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr2),
                                                        columns=dataset_orig.columns)
        dataset_orig_test_3 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr3)
        dataset_orig_train_new_for_attr3 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr3),
                                                        columns=dataset_orig.columns)
        dataset_orig_test_4 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)


        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                                 protected_attribute_names=macro_var[dataset_used])
        dataset_orig_train_new_for_attr1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr1,
                                                label_names=['Probability'],
                                                protected_attribute_names=[macro_var[dataset_used][0]])
        dataset_orig_train_new_for_attr2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr2,
                                                    label_names=['Probability'],
                                                    protected_attribute_names=[macro_var[dataset_used][1]])
        dataset_orig_train_new_for_attr3 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                                              df=dataset_orig_train_new_for_attr3,
                                                              label_names=['Probability'],
                                                              protected_attribute_names=[macro_var[dataset_used][2]])
        dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                                label_names=['Probability'],
                                                protected_attribute_names=macro_var[dataset_used])
        dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[macro_var[dataset_used][0]])
        dataset_orig_test_3 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_3,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[macro_var[dataset_used][1]])
        dataset_orig_test_4 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_4,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[macro_var[dataset_used][2]])

        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator = clf)
        clf1 = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)

        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator = clf)
        clf2 = clf.fit(dataset_orig_train_new_for_attr1.features, dataset_orig_train_new_for_attr1.labels)

        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf3 = clf.fit(dataset_orig_train_new_for_attr2.features, dataset_orig_train_new_for_attr2.labels)

        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf4 = clf.fit(dataset_orig_train_new_for_attr3.features, dataset_orig_train_new_for_attr3.labels)

        test_df_copy = copy.deepcopy(dataset_orig_test_1)
        pred_de1 = clf1.predict_proba(dataset_orig_test_1.features)
        pred_de2 = clf2.predict_proba(dataset_orig_test_2.features)
        pred_de3 = clf3.predict_proba(dataset_orig_test_3.features)
        pred_de4 = clf4.predict_proba(dataset_orig_test_4.features)

        res = []
        for i in range(len(pred_de1)):
            prob_t = (pred_de1[i][1]+pred_de2[i][1]+pred_de3[i][1]+pred_de4[i][1])/4
            if prob_t >= 0.5:
                res.append(1)
            else:
                res.append(0)

        test_df_copy.labels = np.array(res).reshape(-1,1)

        round_result= measure_final_score(dataset_orig_test_1,test_df_copy,macro_var[dataset_used])
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index)
        for i in range(repeat_time):
            fout.write('\t%f' % results[p_index][i])
        fout.write('\n')
    fout.close()
