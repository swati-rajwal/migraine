from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import sklearn
import pandas as pd
import io
import statistics as stats
import sys, glob
import data_utils
from eval_metric import evaluate
import bootstrap_ci
import json

class TaskModel:
    def __init__(self, configs):

        # required
        self.classes = configs['classes']   # list of classes
        self.nfold = configs['nfold']       # list of fold ID
        self.metric = configs['metric']
        self.model = configs['model']           # the name used to name the output folder
        self.proj = configs['proj']             # the name used to name the input folder 

        # optional
        self.pos_label = configs['pos_label'] if 'pos_label' in configs else None
        self.metric_ci = configs['metric_ci'] if 'metric_ci' in configs else None # the metric used for computing CI
        self.merge_pred= configs['merge_pred'] if 'merge_pred' in configs else None # the metric used for computing CI
        self.pred_file = configs['pred_file'] if 'pred_file' in configs else None  # the prediction file name
        self.sub_model = configs['sub_model'] if 'sub_model' in configs else None


    def get_data_dir(self, i):
        data_dir = f'data/{self.proj}/data_splits_{i}/'
        if self.sub_model:
            data_dir += self.sub_model
        return data_dir


    def get_pred_file_path(self, i):

        if self.model == 'svm':
            model_dir = self.get_data_dir(i)
            if self.pred_file:
                pred_file = f'{model_dir}/{self.pred_file}'
            else:
                pred_file = f'{model_dir}/svm_test_result.txt'

        else:
            model_dir = f'model/{self.proj}/data_splits_{i}_{self.model}'
            if self.sub_model:
                model_dir = f'{model_dir}/{self.sub_model}'

            if self.pred_file:
                pred_file = f'{model_dir}/{self.pred_file}'
            else:
                pred_file = f'{model_dir}/preds.csv'

        return pred_file


def run_eval(configs):
    print('##', configs['proj'])
    
    # task specific variables
    task_model = TaskModel(configs)
    
    # evaluate nfolds
    res = {'f':[], 'p':[], 'r':[]}
    all_labels = []
    all_preds = []
    for n in task_model.nfold:
        data_dir = task_model.get_data_dir(n)
        pred_file = task_model.get_pred_file_path(n)
    
        if task_model.model == 'svm':
            pred_df = data_utils.load_data(pred_file, sep='\t', label_name='prediction', text_name='index')
        else:
            pred_df = data_utils.load_data(pred_file, sep='\t', label_name='prediction', text_name='index',
                                        classes=task_model.classes, reverse=True)
    
        label_file = f'{data_dir}/test.csv'
        label_df = data_utils.load_data(label_file)
    
        labels = label_df.labels
        preds = pred_df.labels
    
        all_labels.append(labels)
        all_preds.append(preds)
    
        p, r, f = evaluate(preds=preds, labels=labels, metric=task_model.metric, pos_label=task_model.pos_label)
    
        f = round(f, 3)
        p = round(p, 3)
        r = round(r, 3)
    
        res['f'].append(f)
        res['p'].append(p)
        res['r'].append(r)
    
        # print classification report
        report = classification_report(y_true=labels, y_pred=preds)
        print(report)
    
    
    if len(res['f']) > 1:
        print('========================Eval average=============================')
        avg_f1 = stats.mean(res['f'])
        avg_p = stats.mean(res['p'])
        avg_r = stats.mean(res['r'])
        std_f1 = stats.stdev(res['f'])
        std_p = stats.stdev(res['p'])
        std_r = stats.stdev(res['r'])
        print(f'{avg_p:.2f} (±{std_p:.2f})\t{avg_r:.2f} (±{std_r:.2f})\t{avg_f1:.2f} (±{std_f1:.2f})')
    
    if task_model.metric_ci:
        if task_model.merge_pred:
            print('\n========================Eval merged preds with 95% CI=============================')
            all_preds_list = []
            all_labels_list = []
            for i in range(len(all_preds)):
                for j in range(len(all_preds[i])):
                    all_preds_list.append(all_preds[i][j]) 
                    all_labels_list.append(all_labels[i][j]) 
            p, r, f1 = evaluate(preds=all_preds_list, labels=all_labels_list, metric=task_model.metric, pos_label=task_model.pos_label) 
            lower, upper = bootstrap_ci.process(all_preds_list, all_labels_list, task_model.metric_ci, task_model.pos_label)
            print(classification_report(y_true=all_labels_list, y_pred=all_preds_list))

        else:
            print('\n========================Eval median with 95% CI=============================')
            if len(res['f']) == 1:
                f1 = res['f'][0]
                p = res['p'][0]
                r = res['r'][0]
                split_no = 0
            else:
                f1 = stats.median_high(res['f'])
                split_no = res['f'].index(f1)
                p = res['p'][split_no]
                r = res['r'][split_no]
                print('Median split No based on f1:', split_no+1)
    
            # compute the 95% CI on the best split
            lower, upper = bootstrap_ci.process(all_preds[split_no], all_labels[split_no], task_model.metric_ci, task_model.pos_label)

        print((p, r, f1, lower, upper))
        #print('##\t{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}-{4:.2f}'.format(p, r, f1, lower, upper))
        print(f'##\t{p:.2f} ({lower[0]:.2f}-{upper[0]:.2f})\t{r:.2f} ({lower[1]:.2f}-{upper[1]:.2f})\t{f1:.2f} ({lower[2]:.2f}-{upper[2]:.2f})')

        return p, r, f1, lower, upper


if __name__ == '__main__':

    config_file = sys.argv[1]
    with open(config_file) as f:
        configs = json.load(f)
    run_eval(configs)

    #config_file = 'task_configs/mental_health_configs.json'
    #out_file = 'Aug_mental_health_gpt-4_result.csv'


    #resutls = []
    #with open(config_file) as f:
    #    configs = json.load(f)

    #n = 5
    #for i in range(2, n+2):
    #    prev = i - 1
    #    for j in range(10, 110, 10):
    #        configs['proj'] = f'MH_aug_data_gpt4_{prev}/Aug_human_percent_{j}'
    #        p, r, f1, lower, upper = run_eval(configs)
    #        resutls.append([prev, j, p, r, f1, lower, upper])

    #out_df = pd.DataFrame(resutls, columns=['n_post', 'percent', 'precision', 'recall', 'f1', 'lower', 'upper'])
    #out_df.to_csv(out_file, index=False)
