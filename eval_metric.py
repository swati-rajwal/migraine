import sys
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pdb
import data_utils

def read_csv(input_file, sep=','):
    try:
        df = pd.read_csv(input_file, sep=sep)
    except:
        try:
            df = pd.read_csv(input_file, sep=sep, lineterminator='\n')
        except:
            raise
    return df


def evaluate(preds, labels, metric, pos_label=None):
    res = None

    if metric == 'acc' :
        res = accuracy_score(preds, labels)

    elif metric == 'f1_macro_weighted' :
        res = f1_score(y_true=labels, y_pred=preds, average='weighted')

    elif metric == 'f1_macro' :
        res = f1_score(y_true=labels, y_pred=preds, average='macro')

    elif metric == 'f1_micro' :
        res = f1_score(y_true=labels, y_pred=preds, average='micro')

    elif metric == 'pos_class_f1' :
        return f1_score(y_true=labels, y_pred=preds, pos_label=pos_label)

    elif metric == 'pos_class' :
        #res = f1_score(y_true=labels, y_pred=preds)
        #print(classification_report(y_true=labels, y_pred=preds))
        f = f1_score(y_true=labels, y_pred=preds, pos_label=pos_label)
        p = precision_score(y_true=labels, y_pred=preds, pos_label=pos_label)
        r = recall_score(y_true=labels, y_pred=preds, pos_label=pos_label)
        return p, r, f

    elif metric == 'neg_class_f1' :
        res = f1_score(y_true=labels, y_pred=preds, pos_label=0)

    elif metric == 'f1_pmabuse' :
        cls_repo = classification_report(y_true=labels, y_pred=preds, output_dict=True)
        res = cls_repo['0']['f1-score']

    elif metric == 'f1_report' :
        cls_repo = classification_report(y_true=labels, y_pred=preds)
        print(cls_repo)

    elif metric == 'f1_report_dict' :
        cls_repo = classification_report(y_true=labels, y_pred=preds, output_dict=True)
        res = '{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}'.format(cls_repo['1']['precision'], cls_repo['1']['recall'], cls_repo['1']['f1-score'], cls_repo['accuracy'])

    elif metric == 'cls_specific_class' :
        cls_repo = classification_report(y_true=labels, y_pred=preds, output_dict=True)
        #print(cls_repo)
        f1 = cls_repo[pos_label]['f1-score']
        p = cls_repo[pos_label]['precision']
        r = cls_repo[pos_label]['recall']
        return p, r, f1

    elif metric == 'micro':
        f1 = f1_score(labels, preds, average='micro')
        p = precision_score(labels, preds, average='micro')
        r = recall_score(labels, preds, average='micro')
        return p, r, f1

    return res


if __name__ == '__main__':
    pred_file = sys.argv[1]
    label_file = sys.argv[2]
    metric = sys.argv[3]
    classes=['Fontan', 'NotFontan']

    label_df = data_utils.load_data(label_file)
    pred_df = data_utils.load_data(pred_file, sep='\t', label_name='prediction', text_name='index',
                                    classes=classes, reverse=True)
    assert len(label_df) == len(pred_df), 'gold:{}, pred:{}'.format(len(label_df), len(pred_df))

    labels = label_df.labels
    preds = pred_df.labels

    res = evaluate(preds, labels, metric)
    print('{}'.format(res))
