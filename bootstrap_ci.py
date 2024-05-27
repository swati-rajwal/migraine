from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from eval_metric import evaluate
import pdb
import pandas as pd
import data_utils

def process(preds, labels, metric, pos_label=None):
    pred_gold_pairs = {}
    for i, (pred, label) in enumerate(zip(preds, labels)):
        pred_gold_pairs[i] = [label, pred]

    #PREPARE FOR 1000 SIMULATONS
    simulations = []
    #num_samples = 20000
    num_samples = 1000
    sample_size = len(pred_gold_pairs.keys())
    #sample_size = 100
    for c in range(num_samples):
        #random choosing with replacement
        itersample = np.random.choice(list(pred_gold_pairs.keys()), size = sample_size, replace = True)
        simulations.append(itersample)

    #print (len(simulations))
    #pdb.set_trace()
    score_per_simul = []
    for s in simulations:
        actual = []
        pred = []
        for s_key in s:
            actual.append(pred_gold_pairs[s_key][0])
            pred.append(pred_gold_pairs[s_key][1])
        score = evaluate(pred, actual, metric, pos_label)
        score_per_simul.append(score)

    if type(score_per_simul[0]) == list and len(score_per_simul[0]) > 0: 
        n_metrics = len(score_per_simul[0])
        lower = []
        upper = []
        for k in range(n_metrics):
            score_per_simul_metric = [x[k] for x in score_per_simul]
            score_per_simul_metric.sort()
            lower.append(score_per_simul_metric[int(num_samples * 0.025) - 1])
            upper.append(score_per_simul_metric[int(num_samples * 0.975) - 1])
    else:
        #sort the results
        score_per_simul.sort()

        lower = score_per_simul[int(num_samples * 0.025) - 1]
        upper = score_per_simul[int(num_samples * 0.975) - 1]
    #print('Summary statistics:')
    #return '{0:.1f}\t{1:.1f}\t{2:.1f}\t{3:.1f}'.format(score_per_simul[int(num_samples/2)]*100, lower*100, upper*100, (upper-lower)*100/2)
    #return '{0:.1f}±{1:.1f}'.format(score_per_simul[int(num_samples/2)]*100, (upper-lower)*100/2)
    return lower, upper


if __name__ == '__main__':
    #preds = [1, 1, 0, 0, 1]
    #labels = [0, 0, 0, 0, 1]
    import sys
    pred_file = sys.argv[1]
    label_file = sys.argv[2]
    metric = 'pos_class_f1'
    pos_label = 'Fontan'
    classes = ['Fontan', 'NotFontan']

    label_df = data_utils.load_data(label_file)
    pred_df = data_utils.load_data(pred_file, sep='\t', 
            label_name='prediction', text_name='index', classes=classes, reverse=True)

    labels = label_df.labels
    preds = pred_df.labels

    print(preds)
    print(labels)
    lower, upper = process(preds, labels, metric, pos_label)
    print(f'{lower:.2f}-{upper:.2f}')