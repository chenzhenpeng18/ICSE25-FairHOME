import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Fairea_multi_Three.utility import get_data
from Fairea_multi_Three.fairea_dl import create_baseline
import argparse
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['compas_new'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['dl'], help="Classifier name")

args = parser.parse_args()
dataset_used = args.dataset
clf_name = args.clf

attr_dict = {"compas_new":["sex","race","age"]}

degrees = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
dataset_orig,  mutation_strategies = get_data(dataset_used)

fout = open(dataset_used+'_'+clf_name+'_baseline','w')
res = create_baseline(clf_name,dataset_orig,
                    data_splits=20,repetitions=20,odds=mutation_strategies,options = [0,1],
                   degrees = degrees,p_attrs=attr_dict[dataset_used])
per_list = ['accuracy', 'precision', 'recall', 'f1score', 'mcc']
fair_list = ['wcspd', 'wcaod','wceod', 'avespd', 'aveaod','aveeod']

res_forplot = {}
index = '0'
if dataset_used == 'compas_new':
    index = '1'
for i in range(len(per_list)):
    res_forplot[per_list[i]]= {}
    res_forplot[per_list[i]][index] = np.array([np.mean([row[6+i] for row in res[index][degree]]) for degree in degrees])

for i in range(len(fair_list)):
    res_forplot[fair_list[i]]= {}
    res_forplot[fair_list[i]][index] = np.array([np.mean([row[i] for row in res[index][degree]]) for degree in degrees])

for i in per_list:
    fout.write(i)
    for numnum in res_forplot[i][index]:
        fout.write('\t%f' % numnum)
    fout.write('\n')

for i in fair_list:
    fout.write(i)
    for numnum in res_forplot[i][index]:
        fout.write('\t%f' % numnum)
    fout.write('\n')

fout.close()
