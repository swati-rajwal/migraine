#!/usr/bin/env python
# coding: utf-8
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from data_utils import load_data, get_classes
import os

import argparse
import ast
import numpy as np
import pandas as pd 

parser = argparse.ArgumentParser(description='Process model parameters')

# data related parameters
parser.add_argument('--data_dir', type=str, required=True, help='The directory of the train, dev, and test set.')
parser.add_argument('--out_dir', type=str, required=True, help='The directory of the output model/checkpoints and predictions')

# model name
parser.add_argument('--model_type', type=str, required=True, help='For example, roberta is the model type of roberta-base')
parser.add_argument('--model_name', type=str, required=True, help='The specific model name or model path')

# model hyper-parameters
parser.add_argument('--epoch', type=int, required=True, help='the number of epochs')
parser.add_argument('--sliding_window', type=bool, default=False, help='use sliding window or not')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--max_seq_length', type=int, default=512)

# environment
parser.add_argument('--n_gpu', type=int, default=1)

# other
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--do_predict', action='store_true')
parser.add_argument('--do_predict_prob', action='store_true')


args = parser.parse_args()
data_dir = args.data_dir
out_dir = args.out_dir
pred_out_file = f'{out_dir}/preds.csv'

#classes= ast.literal_eval(args.classes)  #['NotFontan', 'Fontan']
classes = get_classes(args.data_dir)
print(f"Number of classes: {classes}",flush=True)
num_label = len(classes)
print(f"Value of num_label: {num_label}",flush=True)

# load data
df_train = load_data(f'{data_dir}/train.csv', is_train=True, classes=classes)
df_val = load_data(f'{data_dir}/dev.csv', is_train=False, classes=classes)

if os.path.exists(f'{data_dir}/test.csv'):
    df_test = load_data(f'{data_dir}/test.csv', is_train=False, classes=classes)
else:
    df_test = load_data(f'{data_dir}/dev.csv', is_train=False, classes=classes)

# model arch
model_args = ClassificationArgs()
model_args.safe_serialization=False
model_args.sliding_window = args.sliding_window
model_args.max_seq_length = args.max_seq_length
model_args.train_batch_size = args.train_batch_size
model_args.eval_batch_size = args.eval_batch_size
model_args.gradient_accumulation_steps = args.gradient_accumulation_steps
model_args.num_train_epochs = args.epoch
model_args.n_gpu = args.n_gpu
# input data process
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
# save checkpoints
model_args.best_model_dir = f'{out_dir}/best_model'
model_args.use_early_stopping = True
model_args.early_stopping_patience = 10
model_args.evaluate_during_training = True
# other
model_args.no_cache = True

# initialize the model
model = ClassificationModel(args.model_type, args.model_name, num_labels=num_label, use_cuda=True, args=model_args) 

# training
if args.do_train:
    print('Run training')
    model.train_model(df_train, eval_df=df_val, output_dir=out_dir)

if args.do_predict:
    print('Run inference')
    # testing the best checkpoint
    model_path=model_args.best_model_dir
    model = ClassificationModel(args.model_type, model_path, num_labels=num_label, use_cuda=True, args=model_args) 
    
    # prediction and eval
    test_text=list(df_test['text'])
    Y_test=df_test['labels']
    predictions, raw_outputs = model.predict(test_text)
    report = classification_report(Y_test, predictions)
    print(report)
    
    with open(pred_out_file, 'w') as fw:
        fw.write('index\tprediction\n')
        for i, pred in enumerate(predictions):
            fw.write(f'{i}\t{pred}\n')
    
if args.do_predict_prob:

    def sigmoid(x):
        x = np.asarray(x)
        return 1.0/(1 + np.exp(-x))


    def strfmt_prob(prob_array):
        ret = []
        n_samples, n_classes = prob_array.shape
        for i in range(n_samples):
            str_prob = ','.join([str(prob_array[i][j]) for j in range(n_classes)])
            ret.append(str_prob)
        return ret


    # testing the best checkpoint
    print('Run inference prob')
    model_path=model_args.best_model_dir
    model = ClassificationModel(args.model_type, model_path, num_labels=num_label, use_cuda=True, args=model_args) 

    # prediction and eval
    test_text=list(df_test['text'])
    Y_test=df_test['labels']
    predictions, raw_outputs = model.predict(test_text)
    print(len(raw_outputs), len(raw_outputs[0])) 

    # get normalized scores
    pred_probs = []
    for raw_output in raw_outputs:
        norm = sigmoid(raw_output[0])
        pred_probs.append(norm)
    pred_probs = np.asarray(pred_probs)
    print(pred_probs.shape)

    df = pd.DataFrame({
            'index':[i for i in range(len(predictions))],
            'prediction':predictions,
            'probas':strfmt_prob(pred_probs)
        })
    df.to_csv(pred_out_file, index=False, sep='\t')

