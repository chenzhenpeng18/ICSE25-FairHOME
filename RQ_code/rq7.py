import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

approach_list = ['rew','adv','eop','fairsmote', 'maat','fairmask','gry', 'fairhome']
task_list = ['adult', 'compas_new', 'default', 'german', 'mep1', 'mep2']
model_list = ['lr','rf','svm','dl']
data = {}
for i in model_list:
    data[i]={}
    for j in task_list:
        data[i][j]={}
        if j == 'compas_new':
            for k in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'spd2', 'aod2', 'eod2']:
                data[i][j][k]={}
        else:
            for k in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1']:
                data[i][j][k]={}


data_key_value_used = {6:'spd0', 7: 'aod0', 8: 'eod0', 9: 'spd1', 10:'aod1', 11: 'eod1'}
for j in model_list:
    for name in ['origin', 'rew', 'eop','fairsmote', 'maat','fairmask','fairhome']:
        for dataset in ['adult', 'default', 'mep1', 'mep2','german']:
            fin = open('../Results/'+name+'_'+j+'_'+dataset +'.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv','gry']:
    for dataset in ['adult', 'default', 'mep1', 'mep2','german']:
        fin = open('../Results/'+name+'_lr_'+dataset +'.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr','rf','svm','dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

data_key_value_used = {6:'spd0', 7: 'aod0', 8: 'eod0', 9: 'spd1', 10:'aod1', 11: 'eod1', 12:'spd2', 13:'aod2', 14:'eod2'}
for j in model_list:
    for name in ['origin', 'rew', 'eop','fairsmote', 'maat','fairmask', 'fairhome']:
        for dataset in ['compas_new']:
            fin = open('../Results_Three/'+name+'_'+j+'_'+dataset +'.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv','gry']:
    for dataset in ['compas_new']:
        fin = open('../Results_Three/'+name+'_lr_'+dataset +'.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr','rf','svm','dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

fout = open('rq7_result', 'w')
fout.write('\tincrease\ttie\tdecrease\n')
for approach in approach_list:
    num_reduce = 0
    num_incre = 0
    num_equal = 0
    for task in task_list:
        for j in model_list:
            for metric in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1']:
                if mann(data[j][task][metric]['origin'], data[j][task][metric][approach]) >= 0.05:
                    num_equal+=1
                else:
                    default_num = mean(data[j][task][metric]['origin'])
                    fh_num = mean(data[j][task][metric][approach])
                    if default_num > fh_num:
                        num_incre += 1
                    if default_num < fh_num:
                        num_reduce += 1
            if task == 'compas_new':
                for metric in ['spd2', 'aod2', 'eod2']:
                    if mann(data[j][task][metric]['origin'], data[j][task][metric][approach]) >= 0.05:
                        num_equal+=1
                    else:
                        default_num = mean(data[j][task][metric]['origin'])
                        fh_num = mean(data[j][task][metric][approach])
                        if default_num > fh_num:
                            num_incre += 1
                        if default_num < fh_num:
                            num_reduce += 1
    fout.write(approach)
    fout.write("\t%d\t%d\t%d\n" % (num_incre, num_equal, num_reduce))

fout.close()
