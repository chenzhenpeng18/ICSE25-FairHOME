import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

approach_list = ['rew','adv','eop','fairsmote', 'maat','fairmask', 'gry', 'fairhome']
task_list = ['adult', 'compas_new', 'default', 'german', 'mep1', 'mep2']
model_list = ['lr','rf','svm','dl']
data = {}
for i in model_list:
    data[i]={}
    for j in task_list:
        data[i][j]={}
        for k in ['accuracy','precision','recall','f1score','mcc', 'wcspd','wcaod','wceod', 'avespd','aveaod','aveeod']:
            data[i][j][k]={}

data_key_value_used = {1:'accuracy', 2: 'recall', 3: 'precision', 4: 'f1score', 5: 'mcc', 12: 'wcspd', 13: 'wcaod', 14: 'wceod', 15: 'avespd', 16: 'aveaod', 17: 'aveeod'}
for j in model_list:
    for name in ['origin', 'rew', 'eop', 'fairsmote', 'maat','fairmask','fairhome']:
        for dataset in ['adult', 'default', 'mep1', 'mep2','german']:
            fin = open('../Results/'+name+'_'+j+'_'+dataset +'.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv', 'gry']:
    for dataset in ['adult', 'default', 'mep1', 'mep2','german']:
        fin = open('../Results/'+name+'_lr_'+dataset +'.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in model_list:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

data_key_value_used = {1:'accuracy', 2: 'recall', 3: 'precision', 4: 'f1score', 5: 'mcc', 15: 'wcspd', 16: 'wcaod', 17: 'wceod', 18: 'avespd', 19: 'aveaod', 20:'aveeod'}
for j in model_list:
    for name in ['origin','rew','eop','fairsmote', 'maat','fairmask','fairhome']:
        for dataset in ['compas_new']:
            fin = open('../Results_Three/'+name+'_'+j+'_'+dataset +'.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv', 'gry']:
    for dataset in ['compas_new']:
        fin = open('../Results_Three/'+name+'_lr_'+dataset +'.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr', 'rf', 'svm','dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

fout = open('rq1_result','w')
fout.write("-------Results for TableII\n")
for task in task_list:
    fout.write("\\hline\n\\multirow{2}*{"+task+"}")
    for name in ['origin', 'fairhome']:
        fout.write('&'+name)
        for j in ['lr', 'rf']:
            for metric in ['wcspd','wcaod','wceod','avespd','aveaod','aveeod','accuracy','precision','recall','f1score','mcc']:
                fout.write('&%.3f' % mean(data[j][task][metric][name]))
        fout.write('\\\\\n')

for task in task_list:
    fout.write("\\hline\n\\multirow{2}*{"+task+"}")
    for name in ['origin', 'fairhome']:
        fout.write('&'+name)
        for j in ['svm', 'dl']:
            for metric in ['wcspd','wcaod','wceod','avespd','aveaod','aveeod', 'accuracy','precision','recall','f1score','mcc']:
                fout.write('&%.3f' % mean(data[j][task][metric][name]))
        fout.write('\\\\\n')

fout.write("\n\n-------Results for TableIII\n")
fout.write('\tOriginal\tFairHOME\tabsc\trelac\n')
for i in ['wcspd','wcaod','wceod','avespd','aveaod','aveeod','accuracy','precision','recall','f1score','mcc']:
    fout.write(i)
    olist = []
    flist = []
    for dataset in task_list:
        for j in model_list:
            originresult = mean(data[j][dataset][i]['origin'])
            fairhomeresult = mean(data[j][dataset][i]['fairhome'])
            olist.append(originresult)
            flist.append(fairhomeresult)
    abschange = mean(flist)-mean(olist)
    relachange = 100*(abschange)/mean(olist)
    fout.write('\t%.3f\t%.3f\t%.3f\t%.1f%%\n' % (mean(olist),mean(flist),abschange,relachange))

fout.close()



