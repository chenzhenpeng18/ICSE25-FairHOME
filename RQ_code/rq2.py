import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

approach_list = ['rew','adv','eop','fairsmote', 'maat','fairmask', 'fairhome','gry']
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

fout = open('rq2_result','w')
fout.write("-------Results for TableIV\n")
fout.write('\twcspdabsc\twcspdrelac\twcaodabsc\twcaodrelac\twceodabsc\twceodrelac\tacspdabsc\tacspdrelac\tacaodabsc\tacaodrelac\taceodabsc\taceodrelac\n')
for name in ['rew', 'adv','eop','fairsmote', 'maat','fairmask','gry','fairhome']:
    fout.write(name)
    for i in ['wcspd','wcaod','wceod', 'avespd','aveaod','aveeod']:
        olist = []
        nlist = []
        for dataset in task_list:
            for j in model_list:
                olist.append(mean(data[j][dataset][i]['origin']))
                nlist.append(mean(data[j][dataset][i][name]))
        abschange = mean(nlist) - mean(olist)
        relachange = 100 * (mean(nlist) - mean(olist)) / mean(olist)
        fout.write('\t%.3f\t%.1f%%' % (abschange, relachange))
    fout.write('\n')

fout.write("\n\n-------Results for TableV\n")
for name in ['rew', 'adv','eop','fairsmote', 'maat','fairmask','gry']:
    fout.write('\t'+name)
fout.write('\n')

count = {}
for name in ['rew', 'adv', 'eop', 'fairsmote', 'maat', 'fairmask', 'gry']:
    count[name] = {}
    for i in ['wcspd','wcaod','wceod', 'avespd','aveaod','aveeod']:
        count[name][i]={}
        for typee in ['win','tie','lose']:
            count[name][i][typee] = 0

for i in ['wcspd','wcaod','wceod', 'avespd','aveaod','aveeod']:
    fout.write(i)
    for name in ['rew', 'adv', 'eop', 'fairsmote', 'maat', 'fairmask', 'gry']:
        for dataset in ['adult', 'default', 'mep1', 'mep2', 'compas_new', 'german']:
            for j in model_list:
                if mann(data[j][dataset][i][name], data[j][dataset][i]['fairhome']) >= 0.05:
                    count[name][i]['tie'] += 1
                elif mean(data[j][dataset][i][name]) > mean(data[j][dataset][i]['fairhome']):
                    count[name][i]['win'] += 1
                else:
                    count[name][i]['lose'] += 1
        fout.write('\t'+str(count[name][i]['win'])+'/'+str(count[name][i]['tie'])+'/'+str(count[name][i]['lose']))
    fout.write('\n')

fout.write('Overall')
for name in ['rew', 'adv', 'eop', 'fairsmote', 'maat', 'fairmask', 'gry']:
    sumlist={}
    sumlist['win'] = 0
    sumlist['tie'] = 0
    sumlist['lose'] = 0
    for i in ['wcspd', 'wcaod', 'wceod', 'avespd', 'aveaod', 'aveeod']:
        sumlist['win'] += count[name][i]['win']
        sumlist['tie'] += count[name][i]['tie']
        sumlist['lose'] += count[name][i]['lose']
    fout.write('\t'+str(sumlist['win'])+'/'+str(sumlist['tie'])+'/'+str(sumlist['lose']))
fout.write('\n')

fout.close()
