import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score, matthews_corrcoef
from aif360.metrics import ClassificationMetric
from numpy import mean, abs

def cal_spd(dataset_test_pred, p_attr):
    labelname = 'Probability'
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    num1 = len(dataset_test_pred[(dataset_test_pred[p_attr] == 0)  & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[p_attr] == 0)])
    num2 = len(dataset_test_pred[(dataset_test_pred[p_attr] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[p_attr] == 1)])
    return [max([num1,num2])-min([num1,num2])]

def cal_eod(dataset_test, dataset_test_pred, p_attr):
    labelname = 'Probability'
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    num1 = len(dataset_test[(
            dataset_test[p_attr] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[
        labelname] == 1)])
    num2 = len(dataset_test[(
            dataset_test[p_attr] == 1)& (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[
            labelname] == 1)])

    return [max([num1,num2]) - min([num1,num2])]

def cal_aod(dataset_test, dataset_test_pred, p_attr):
    labelname = 'Probability'
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    num1 = len(dataset_test[(
            dataset_test[p_attr] == 0) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[p_attr] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 0)  & (dataset_test[
            labelname] == 1)])
    num2 = len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 0)]) + len(dataset_test[(dataset_test[p_attr] == 1) & (                                                                                                         dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1)])
    return [(max([num1,num2]) - min([num1,num2]))/2]

def wc_spd(dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = 1
    labelname = 'Probability'
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test_pred[labelname] = np.where(dataset_test_pred[labelname] == favorlabel, 1, 0)
    num1 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0)])
    num2 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1)])
    num3 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0)])
    num4 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1)])
    return [max([num1,num2,num3,num4])-min([num1,num2,num3,num4])]

def ave_spd(dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = 1
    labelname = 'Probability'
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test_pred[labelname] = np.where(dataset_test_pred[labelname] == favorlabel, 1, 0)
    num_average = len(dataset_test_pred[dataset_test_pred[labelname] == 1]) / len(dataset_test_pred)
    num1 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0)])
    num2 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1)])
    num3 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0)])
    num4 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1)])
    return [mean([abs(num1-num_average),abs(num2-num_average),abs(num3-num_average),abs(num4-num_average)])]

def wc_aod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = dataset_test.favorable_label
    labelname = dataset_test.label_names[0]
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred'+labelname] = np.where(dataset_test['pred'+labelname] == favorlabel, 1, 0)
    num_list = []
    num1 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 1)])
    num_list.append(num1)
    num2 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])
    num_list.append(num2)
    num3 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 1)])
    num_list.append(num3)
    num4 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])
    num_list.append(num4)
    return [(max(num_list) - min(num_list))/2]

def ave_aod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = dataset_test.favorable_label
    labelname = dataset_test.label_names[0]
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred'+labelname] = np.where(dataset_test['pred'+labelname] == favorlabel, 1, 0)

    num_average = len(dataset_test[(dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(dataset_test[(dataset_test[labelname] == 0)]) + len(dataset_test[(dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(dataset_test[(dataset_test[labelname] == 1)])

    num1 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 1)])

    num2 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])

    num3 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 1)])

    num4 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])

    return [mean([abs(num1 - num_average), abs(num2 - num_average), abs(num3 - num_average), abs(num4 - num_average)])/2]

def wc_eod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = 1
    labelname = 'Probability'
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred' + labelname] = np.where(dataset_test['pred' + labelname] == favorlabel, 1, 0)
    num_list=[]
    # if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) != 0:
    num1 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1)])
    num_list.append(num1)
    # if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) != 0:
    num2 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])
    num_list.append(num2)
    # if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) != 0:
    num3 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 1)])
    num_list.append(num3)
    # if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) != 0:
    num4 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])
    num_list.append(num4)
    return [max(num_list) - min(num_list)]

def ave_eod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = 1
    labelname = 'Probability'
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred' + labelname] = np.where(dataset_test['pred' + labelname] == favorlabel, 1, 0)
    num_average = len(dataset_test[(dataset_test[labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(dataset_test[(dataset_test[labelname] == 1)])
    num1 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1)])
    num2 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])
    num3 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[
            labelname] == 1)])
    num4 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[
            labelname] == 1)])

    return [mean([abs(num1-num_average),abs(num2-num_average),abs(num3-num_average),abs(num4-num_average)])]

def measure_final_score(dataset_orig_test, dataset_orig_predict, p_attrs):
    y_test = np.array(dataset_orig_test.labels).flatten()
    y_pred = np.array(dataset_orig_predict.labels).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    TP = np.sum((y_test == 1) & (y_pred == 1))
    FN = np.sum((y_test == 1) & (y_pred == 0))

    return [accuracy, recall_macro,  precision_macro,  f1score_macro, mcc]+ cal_spd(dataset_orig_predict, p_attrs[0]) + cal_aod(dataset_orig_test, dataset_orig_predict, p_attrs[0]) + cal_eod(dataset_orig_test, dataset_orig_predict, p_attrs[0]) + cal_spd(dataset_orig_predict, p_attrs[1]) + cal_aod(dataset_orig_test, dataset_orig_predict, p_attrs[1]) + cal_eod(dataset_orig_test, dataset_orig_predict, p_attrs[1]) + wc_spd(dataset_orig_predict, p_attrs) + wc_aod(dataset_orig_test, dataset_orig_predict, p_attrs) + wc_eod(dataset_orig_test, dataset_orig_predict, p_attrs) + ave_spd(dataset_orig_predict, p_attrs) + ave_aod(dataset_orig_test, dataset_orig_predict, p_attrs) + ave_eod(dataset_orig_test, dataset_orig_predict, p_attrs)
