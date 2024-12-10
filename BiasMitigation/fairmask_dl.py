import pandas as pd
import numpy as np
import copy,os
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from Measure_new import measure_final_score
import argparse
import tensorflow as tf
from aif360.datasets import BinaryLabelDataset
from utility import get_classifier

def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['adult', 'default', 'mep1', 'mep2','german'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['dl'], help="Classifier name")

    args = parser.parse_args()
    dataset_used = args.dataset
    clf_name = args.clf

    macro_var = {'adult': ['sex','race'], 'default':['sex','age'], 'mep1': ['sex','race'], 'mep2': ['sex','race'],'german': ['sex','age']}

    multi_attr = macro_var[dataset_used]

    val_name = "fairmask_{}_{}.txt".format(clf_name, dataset_used)
    fout = open(val_name, 'w')

    results = {}
    performance_index = ['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0', 'aod0', 'eod0', 'spd1',
                         'aod1', 'eod1', 'wcspd', 'wcaod', 'wceod', 'avespd', 'aveaod', 'aveeod']
    for p_index in performance_index:
        results[p_index] = []

    dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()

    repeat_time = 20
    for i in range(repeat_time):
        print(i)
        np.random.seed(i)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
        y_train = copy.deepcopy(dataset_orig_train['Probability'])
        X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
        y_test = copy.deepcopy(dataset_orig_test['Probability'])

        reduced = list(X_train.columns)
        reduced.remove(multi_attr[0])
        reduced.remove(multi_attr[1])

        X_reduced, y_reduced0, y_reduced1 = X_train.loc[:, reduced], X_train[multi_attr[0]],  X_train[multi_attr[1]]
        # Build model to predict the protect attribute0
        clf1_0 = DecisionTreeRegressor()
        sm = SMOTE()
        X_trains, y_trains0 = sm.fit_resample(X_reduced, y_reduced0)
        clf = get_classifier(clf_name,(X_trains.shape[1],))
        clf.fit(X_trains, y_trains0,epochs=20)
        y_proba = clf.predict(X_trains)
        if isinstance(clf1_0, DecisionTreeClassifier) or isinstance(clf1_0, LogisticRegression):
            clf1_0.fit(X_trains, y_trains0)
        else:
            clf1_0.fit(X_trains, y_proba)

        # Build model to predict the protect attribute1
        clf1_1 = DecisionTreeRegressor()
        sm = SMOTE()
        X_trains, y_trains1 = sm.fit_resample(X_reduced, y_reduced1)
        clf = get_classifier(clf_name,(X_trains.shape[1],))
        clf.fit(X_trains, y_trains1,epochs=20)
        y_proba = clf.predict(X_trains)
        if isinstance(clf1_1, DecisionTreeClassifier) or isinstance(clf1_1, LogisticRegression):
            clf1_1.fit(X_trains, y_trains0)
        else:
            clf1_1.fit(X_trains, y_proba)

        X_test_reduced = X_test.loc[:, reduced]
        protected_pred0 = clf1_0.predict(X_test_reduced)
        protected_pred1 = clf1_1.predict(X_test_reduced)
        if isinstance(clf1_0, DecisionTreeRegressor) or isinstance(clf1_0, LinearRegression):
            protected_pred0 = reg2clf(protected_pred0, threshold=0.5)
            protected_pred1 = reg2clf(protected_pred1, threshold=0.5)

        # Build model to predict the target attribute Y
        clf2 = get_classifier(clf_name,(X_train.shape[1],))
        clf2.fit(X_train, y_train,epochs=20)
        X_test.loc[:, multi_attr[0]] = protected_pred0
        X_test.loc[:, multi_attr[1]] = protected_pred1
        y_pred = clf2.predict_classes(X_test)

        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=macro_var[dataset_used])
        test_df_copy = dataset_orig_test.copy(deepcopy=True)
        test_df_copy.labels = y_pred.reshape(-1, 1)

        round_result = measure_final_score(dataset_orig_test, test_df_copy, multi_attr)

        for i in range(len(performance_index)):
                results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index)
        for i in range(repeat_time):
            fout.write('\t%f' % results[p_index][i])
        fout.write('\n')
    fout.close()
