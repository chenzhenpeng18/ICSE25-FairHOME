import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Fairea_multi.fairea import normalize,classify_region
from shapely.geometry import LineString

approach_list = ['rew','adv','eop','fairsmote', 'maat','fairmask','gry', 'fairhome']
task_list = ['adult', 'compas_new', 'default', 'mep1', 'mep2','german']
model_list = ['lr','rf','svm', 'dl']
base_points = {}
for i in model_list:
    base_points[i]={}
    for j in task_list:
        base_points[i][j]={}
for dataset in task_list:
    for i in model_list:
        fin = open('../Fairea_baseline_new/'+dataset+'_'+i+'_baseline','r')
        for line in fin:
            metric_name = line.strip().split('\t')[0]
            base_points[i][dataset][metric_name] = np.array(list(map(float,line.strip().split('\t')[1:])))
        fin.close()

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

region_count = {}
for dataset in task_list:
    region_count[dataset]={}
    for fairmetric in ['wcspd','wcaod','wceod', 'avespd','aveaod','aveeod']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['accuracy','precision','recall','f1score','mcc']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in model_list:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in approach_list:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in model_list:
    for j in task_list:
        for fairmetric in ['wcspd','wcaod','wceod', 'avespd','aveaod','aveeod']:
            for permetric in ['accuracy','precision','recall','f1score','mcc']:
                for name in approach_list:
                    methods = dict()
                    name_fair20 = data[i][j][fairmetric][name]
                    name_per20 = data[i][j][permetric][name]
                    for count in range(20):
                        methods[str(count)] = (float(name_per20[count]), float(name_fair20[count]))
                    normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                    baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                    mitigation_regions = classify_region(baseline, normalized_methods)
                    for count in mitigation_regions:
                        region_count[j][fairmetric][permetric][i][name][mitigation_regions[count]]+=1


fout = open('rq3_result','w')
fout.write("-------Results for Figure2\n")
fout.write('\twin-win\tgood\tpoor\tinverted\tlose-lose\n')
for name in approach_list:
    fout.write(name)
    final_sum = 0
    final_count = {}
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for fairmetric in ['wcspd','wcaod','wceod', 'avespd','aveaod','aveeod']:
        for permetric in ['accuracy','precision','recall','f1score','mcc']:
            for j in task_list:
                for i in ['rf', 'lr', 'svm','dl']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['win-win','good','bad','inverted','lose-lose']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')
fout.close()
