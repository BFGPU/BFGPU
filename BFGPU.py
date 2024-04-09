import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from Dataset import LabeledTextDataset,TextDataset
from Augmentation import synonym_replacement, back_translate
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from sklearn.metrics import recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import wandb
import yaml
import csv
import random
import numpy as np
import math
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling,LineByLineTextDataset
from transformers import RobertaTokenizer, RobertaForMaskedLM,RobertaForSequenceClassification
import numpy as np
import torch
import nltk
from transformers import MarianMTModel, MarianTokenizer
from nltk.corpus import wordnet
import torch.nn.functional as F

os.environ["WANDB_API_KEY"] = "96f8ce07923b9004d37734a7fcf91371990a32d1"
os.environ["WANDB_MODE"] = "offline"
wandb.login(key="96f8ce07923b9004d37734a7fcf91371990a32d1")
wandb.init(
    project="PU_balance_regular_sst_1",
    entity="ygzwqzd",save_code=True,
)
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--bag_imbalance', type=int, default=1)
parser.add_argument('--max_instance_imbalance', type=int, default=10)
parser.add_argument('--self_epoch', type=int, default=10)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--self_sentence_epoch', type=int, default=10)
parser.add_argument('--ood_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--tro', type=float, default=0.1)
parser.add_argument('--scheduler', type=bool, default=True)
parser.add_argument('--synonyms', type=int, default=1)
parser.add_argument('--lambda_ood', type=float, default=1)
parser.add_argument('--dataset', type=str, default='sst2')
parser.add_argument('--augmentation', type=str, default=None)
parser.add_argument('--model', type=str, default='roberta-base')
parser.add_argument('--use_max',type=bool,default=False)
parser.add_argument('--use_hard',type=bool,default=False)
parser.add_argument('--use_balance',type=bool,default=False)
parser.add_argument('--self_model_path', type=str, default='self_model')
parser.add_argument('--model_path', type=str, default='model')
parser.add_argument('--sentence_self_model_path', type=str, default='sentence_self_model')
parser.add_argument('--sentence_model_path', type=str, default='sentence_model')
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--tgt_lang',type=str,default='fr')
parser.add_argument('--temperature',type=float,default=1.0)
parser.add_argument('--self_passage', type=bool, default=False)
parser.add_argument('--self_sentence', type=bool, default=False)
parser.add_argument('--load_self_passage', type=bool, default=False)
parser.add_argument('--load_self_sentence', type=bool, default=False)
parser.add_argument('--train_passage', type=bool, default=False)
parser.add_argument('--train_sentence', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--hint', type=str, default='')
parser.add_argument('--log_file', type=str, default='')
args = parser.parse_args()

def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])
if args.config is not None:
    over_write_args_from_file(args, args.config)
args_dict=vars(args)

f=open(args.log_file+'.csv', "w", encoding="utf-8")
r = csv.DictWriter(f,['instance_imbalance','seed','acc','recall','neg_recall',
    'avg_acc','precision','f1','auc'])
text_file=open(args.log_file+'.txt', "w", encoding="utf-8")

if args.dataset=='sst2':
    df = pd.read_csv('./data/sst2/train.tsv', sep='\t')
    train_labels = df['label'].tolist()
    train_texts = df['content'].tolist()
    df = pd.read_csv('./data/sst2/test.tsv', sep='\t')
    test_labels = df['label'].tolist()
    test_texts = df['content'].tolist()
    positive_train_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==1]
    negative_train_texts = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==0]
    positive_train_labels = [train_labels[i] for i in range(len(train_texts)) if train_labels[i]==1]
    negative_train_labels = [train_labels[i] for i in range(len(train_texts)) if train_labels[i]==0]
    positive_test_texts = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==1]
    negative_test_texts = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==0]
    positive_test_labels = [test_labels[i] for i in range(len(test_texts)) if test_labels[i]==1]
    negative_test_labels = [test_labels[i] for i in range(len(test_texts)) if test_labels[i]==0]

elif args.dataset=='tamatoes':
    df = pd.read_csv('./data/tamatoes/rotten_tomatoes_critic_reviews.csv')
    _texts = df['review_content'].tolist()
    _labels_str = df['review_type'].tolist()
    texts=[]
    labels_str=[]
    for _ in range(len(_texts)):
        if isinstance(_texts[_],str):
            texts.append(_texts[_])
            labels_str.append(_labels_str[_])
    labels=[1 if labels_str[_]=='Fresh' else 0 for _ in range(len(labels_str))]
    positive_texts = [texts[i] for i in range(len(texts)) if labels[i]==1]
    negative_texts = [texts[i] for i in range(len(texts)) if labels[i]==0]

    positive_labels = [labels[i] for i in range(len(texts)) if labels[i]==1]
    negative_labels = [labels[i] for i in range(len(texts)) if labels[i]==0]

    positive_train_texts=positive_texts[:50000]
    negative_train_texts=negative_texts[:50000]
    positive_train_labels=positive_labels[:50000]
    negative_train_labels=negative_labels[:50000]

    positive_test_texts=positive_texts[50000:100000]
    negative_test_texts=negative_texts[50000:100000]
    positive_test_labels=positive_labels[50000:100000]
    negative_test_labels=negative_labels[50000:100000]

elif args.dataset=='imdb_s':
    df = pd.read_csv('./data/imdb/imdb_labelled.txt', sep='\t')
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    positive_texts = [texts[i] for i in range(len(texts)) if labels[i]==1]
    negative_texts = [texts[i] for i in range(len(texts)) if labels[i]==0]

    positive_labels = [labels[i] for i in range(len(texts)) if labels[i]==1]
    negative_labels = [labels[i] for i in range(len(texts)) if labels[i]==0]

    positive_train_texts=positive_texts[:int(len(positive_texts)*0.8)]
    negative_train_texts=negative_texts[:int(len(negative_texts)*0.8)]
    positive_train_labels=positive_labels[:int(len(positive_labels)*0.8)]
    negative_train_labels=negative_labels[:int(len(negative_labels)*0.8)]

    positive_test_texts=positive_texts[int(len(positive_texts)*0.8):]
    negative_test_texts=negative_texts[int(len(negative_texts)*0.8):]
    positive_test_labels=positive_labels[int(len(positive_labels)*0.8):]
    negative_test_labels=negative_labels[int(len(negative_labels)*0.8):]

elif args.dataset=='yelp_s':
    df = pd.read_csv('./data/imdb/yelp_labelled.txt', sep='\t')
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    positive_texts = [texts[i] for i in range(len(texts)) if labels[i]==1]
    negative_texts = [texts[i] for i in range(len(texts)) if labels[i]==0]

    positive_labels = [labels[i] for i in range(len(texts)) if labels[i]==1]
    negative_labels = [labels[i] for i in range(len(texts)) if labels[i]==0]

    positive_train_texts=positive_texts[:int(len(positive_texts)*0.8)]
    negative_train_texts=negative_texts[:int(len(negative_texts)*0.8)]
    positive_train_labels=positive_labels[:int(len(positive_labels)*0.8)]
    negative_train_labels=negative_labels[:int(len(negative_labels)*0.8)]

    positive_test_texts=positive_texts[int(len(positive_texts)*0.8):]
    negative_test_texts=negative_texts[int(len(negative_texts)*0.8):]
    positive_test_labels=positive_labels[int(len(positive_labels)*0.8):]
    negative_test_labels=negative_labels[int(len(negative_labels)*0.8):]
    
elif args.dataset=='amazon_s':
    df = pd.read_csv('./data/amazon/amazon_cells_labelled.txt', sep='\t')
    texts = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    positive_texts = [texts[i] for i in range(len(texts)) if labels[i]==1]
    negative_texts = [texts[i] for i in range(len(texts)) if labels[i]==0]

    positive_labels = [labels[i] for i in range(len(texts)) if labels[i]==1]
    negative_labels = [labels[i] for i in range(len(texts)) if labels[i]==0]

    positive_train_texts=positive_texts[:int(len(positive_texts)*0.8)]
    negative_train_texts=negative_texts[:int(len(negative_texts)*0.8)]
    positive_train_labels=positive_labels[:int(len(positive_labels)*0.8)]
    negative_train_labels=negative_labels[:int(len(negative_labels)*0.8)]

    positive_test_texts=positive_texts[int(len(positive_texts)*0.8):]
    negative_test_texts=negative_texts[int(len(negative_texts)*0.8):]
    positive_test_labels=positive_labels[int(len(positive_labels)*0.8):]
    negative_test_labels=negative_labels[int(len(negative_labels)*0.8):]
    
elif args.dataset=='sentiment140':
    df = pd.read_csv('./data/sentiment140/training.1600000.processed.noemoticon.csv',encoding="ISO-8859-1")
    texts = df.iloc[:, 5]
    labels = df.iloc[:, 0]
    positive_texts = [texts[i] for i in range(len(texts)) if labels[i]==4]
    negative_texts = [texts[i] for i in range(len(texts)) if labels[i]==0]
    positive_labels = [1 for i in range(len(texts)) if labels[i]==4]
    negative_labels = [0 for i in range(len(texts)) if labels[i]==0]
    positive_train_texts=positive_texts[:50000]
    negative_train_texts=negative_texts[:50000]
    positive_train_labels=positive_labels[:50000]
    negative_train_labels=negative_labels[:50000]

    positive_test_texts=positive_texts[50000:100000]
    negative_test_texts=negative_texts[50000:100000]
    positive_test_labels=positive_labels[50000:100000]
    negative_test_labels=negative_labels[50000:100000]

elif args.dataset=='yelp':
    dataset = load_dataset('yelp_polarity')
    train_texts,train_labels = dataset['train']['text'],dataset['train']['label']
    test_texts,test_labels = dataset['test']['text'],dataset['test']['label']
    positive_train_texts, positive_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==1],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==1]
    negative_train_texts, negative_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==0],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==0]
    positive_test_texts, positive_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==1],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==1]
    negative_test_texts, negative_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==0],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==0]

elif args.dataset=='imdb':
    dataset = load_dataset('imdb',data_dir='~/.cache/huggingface/datasets/imdb')
    train_texts,train_labels = dataset['train']['text'],dataset['train']['label']
    test_texts,test_labels = dataset['test']['text'],dataset['test']['label']
    positive_train_texts, positive_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==1],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==1]
    negative_train_texts, negative_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==0],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==0]
    positive_test_texts, positive_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==1],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==1]
    negative_test_texts, negative_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==0],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==0]
    
elif args.dataset=='amazon':
    dataset = load_dataset('amazon_polarity')
    train_texts,train_labels = dataset['train']['content'],dataset['train']['label']
    test_texts,test_labels = dataset['test']['content'],dataset['test']['label']
    positive_train_texts, positive_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==1],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==1]
    negative_train_texts, negative_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==0],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==0]
    positive_test_texts, positive_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==1],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==1]
    negative_test_texts, negative_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==0],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==0]
    positive_train_texts=positive_train_texts[:50000]
    negative_train_texts=negative_train_texts[:50000]
    positive_train_labels=positive_train_labels[:50000]
    negative_train_labels=negative_train_labels[:50000]
    positive_test_texts=positive_test_texts[50000:100000]
    negative_test_texts=negative_test_texts[50000:100000]
    positive_test_labels=positive_test_labels[50000:100000]
    negative_test_labels=negative_test_labels[50000:100000]

print('ptrain:',len(positive_train_texts))
print('ntrain:',len(negative_train_texts))
print('ptest:',len(positive_test_texts))
print('ntest:',len(negative_test_texts))
device = args.device
bag_imbalance=args.bag_imbalance
max_instance_imbalance=args.max_instance_imbalance
model_name=args.model
epoch=args.epoch
batch_size = args.batch_size
weight_decay=args.weight_decay
use_max=args.use_max
use_hard=args.use_hard
use_balance=args.use_balance
tro=args.tro
lr=args.lr

def valid_instance(sentence_model,valid_sentence_dataloader,device,valid_y,V_ind):
    sentence_model.eval()
    sentence_preds = []
    sentence_probability=[]
    scores=[]
    for batch in valid_sentence_dataloader:
        idx,encoded_texts = batch
        encoded_texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in encoded_texts.items()}
        with torch.no_grad():
            outputs = sentence_model(**encoded_texts)
            sentence_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist())
            sentence_probability.extend(nn.Softmax(dim=1)(outputs.logits)[:,0].cpu().numpy().tolist())

    cur_ind=0
    curlabel=1
    pred_y=[]
    for _ in range(len(V_ind)):
        if cur_ind==V_ind[_]:
            curlabel=min(curlabel,sentence_preds[_])
        else:
            pred_y.append(curlabel)
            cur_ind+=1
            curlabel=sentence_preds[_]

    pred_y.append(curlabel)
    instance_acc = accuracy_score(valid_y, pred_y)
    instance_recall = recall_score(valid_y, pred_y)
    instance_neg_recall = recall_score(valid_y, pred_y, pos_label=0)
    instance_avg_acc=(instance_recall+instance_neg_recall)/2
    instance_precision = precision_score(valid_y, pred_y)
    instance_f1 = f1_score(valid_y, pred_y)
    instance_confusion_matrix=confusion_matrix(valid_y, pred_y)
    instance_classification_report=classification_report(valid_y, pred_y)
    print('instance_acc')
    print(instance_acc)
    print('instance_recall')
    print(instance_recall)
    print('instance_neg_recall')
    print(instance_neg_recall)
    print('instance_avg_acc')
    print(instance_avg_acc)
    print('instance_precision')
    print(instance_precision)
    print('instance_f1')
    print(instance_f1)
    print('instance_confusion_matrix')
    print(instance_confusion_matrix)
    print('instance_classification_report')
    print(instance_classification_report)
    args_dict['instance_acc']=instance_acc
    args_dict['instance_recall']=instance_recall
    args_dict['instance_neg_recall']=instance_neg_recall
    args_dict['instance_avg_acc']=instance_avg_acc
    args_dict['instance_precision']=instance_precision
    args_dict['instance_f1']=instance_f1

def valid_instance_sort(sentence_model,valid_sentence_dataloader,device,valid_y,V_ind):
    sentence_model.eval()
    sentence_preds = []
    sentence_probability=[]
    scores=[]
    for batch in valid_sentence_dataloader:
        idx,encoded_texts = batch
        encoded_texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in encoded_texts.items()}
        with torch.no_grad():
            outputs = sentence_model(**encoded_texts)
            sentence_probability.extend(nn.Softmax(dim=1)(outputs.logits)[:,1].cpu().numpy().tolist())
    indexed_list = list(enumerate(sentence_probability))
    sorted_probability = sorted(indexed_list, key=lambda x: x[1])
    threshold=[x[1] for x in sorted_probability][len(valid_y)//2]
    for i in range(len(sentence_probability)):
        if sentence_probability[i]<=threshold:
            sentence_preds.append(0)
        else:
            sentence_preds.append(1)

    cur_ind=0
    curlabel=1
    pred_y=[]
    prob=1
    probs=[]
    for _ in range(len(V_ind)):
        if cur_ind==V_ind[_]:
            prob=prob*sentence_probability[i]
            curlabel=min(curlabel,sentence_preds[_])
        else:
            probs.append(prob)
            pred_y.append(curlabel)
            cur_ind+=1
            curlabel=sentence_preds[_]
            prob=sentence_probability[_]
    probs.append(prob)
    pred_y.append(curlabel)
    
    instance_acc = accuracy_score(valid_y, pred_y)
    instance_recall = recall_score(valid_y, pred_y)
    instance_neg_recall = recall_score(valid_y, pred_y, pos_label=0)
    instance_avg_acc=(instance_recall+instance_neg_recall)/2
    instance_precision = precision_score(valid_y, pred_y)
    instance_f1 = f1_score(valid_y, pred_y)
    instance_auc=roc_auc_score(valid_y, probs)
    instance_confusion_matrix=confusion_matrix(valid_y, pred_y)
    instance_classification_report=classification_report(valid_y, pred_y)
    print('instance_acc')
    print(instance_acc)
    print('instance_recall')
    print(instance_recall)
    print('instance_neg_recall')
    print(instance_neg_recall)
    print('instance_avg_acc')
    print(instance_avg_acc)
    print('instance_precision')
    print(instance_precision)
    print('instance_f1')
    print(instance_f1)
    print('instance_auc')
    print(instance_auc)
    print('instance_confusion_matrix')
    print(instance_confusion_matrix)
    print('instance_classification_report')
    print(instance_classification_report)
    args_dict['instance_acc']=instance_acc
    args_dict['instance_recall']=instance_recall
    args_dict['instance_neg_recall']=instance_neg_recall
    args_dict['instance_avg_acc']=instance_avg_acc
    args_dict['instance_precision']=instance_precision
    args_dict['instance_f1']=instance_f1
    args_dict['instance_auc']=instance_auc

def valid_instance_threshold(sentence_model,valid_sentence_dataloader,device,valid_y,V_ind,threshold):
    sentence_model.eval()
    sentence_preds = []
    sentence_probability=[]
    scores=[]
    for batch in valid_sentence_dataloader:
        idx,encoded_texts = batch
        encoded_texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in encoded_texts.items()}
        with torch.no_grad():
            outputs = sentence_model(**encoded_texts)
            sentence_probability.extend(nn.Softmax(dim=1)(outputs.logits)[:,1].cpu().numpy().tolist())
    for i in range(len(sentence_probability)):
        if sentence_probability[i]<=threshold:
            sentence_preds.append(0)
        else:
            sentence_preds.append(1)

    cur_ind=0
    curlabel=1
    pred_y=[]
    prob=1
    probs=[]
    for _ in range(len(V_ind)):
        if cur_ind==V_ind[_]:
            prob=min(prob,sentence_probability[i])
            curlabel=min(curlabel,sentence_preds[_])
        else:
            probs.append(prob)
            pred_y.append(curlabel)
            cur_ind+=1
            curlabel=sentence_preds[_]
            prob=sentence_probability[_]
    probs.append(prob)
    pred_y.append(curlabel)
    instance_acc = accuracy_score(valid_y, pred_y)
    instance_recall = recall_score(valid_y, pred_y)
    instance_neg_recall = recall_score(valid_y, pred_y, pos_label=0)
    instance_avg_acc=(instance_recall+instance_neg_recall)/2
    instance_precision = precision_score(valid_y, pred_y)
    instance_f1 = f1_score(valid_y, pred_y)
    instance_auc=roc_auc_score(valid_y, probs)
    instance_confusion_matrix=confusion_matrix(valid_y, pred_y)
    instance_classification_report=classification_report(valid_y, pred_y)
    print('instance_acc')
    print(instance_acc)
    print('instance_recall')
    print(instance_recall)
    print('instance_neg_recall')
    print(instance_neg_recall)
    print('instance_avg_acc')
    print(instance_avg_acc)
    print('instance_precision')
    print(instance_precision)
    print('instance_f1')
    print(instance_f1)
    print('instance_auc')
    print(instance_auc)
    print('instance_confusion_matrix')
    print(instance_confusion_matrix)
    print('instance_classification_report')
    print(instance_classification_report)
    args_dict['instance_acc']=instance_acc
    args_dict['instance_recall']=instance_recall
    args_dict['instance_neg_recall']=instance_neg_recall
    args_dict['instance_avg_acc']=instance_avg_acc
    args_dict['instance_precision']=instance_precision
    args_dict['instance_f1']=instance_f1
    args_dict['instance_auc']=instance_auc

def valid_instance_logits(sentence_model,valid_sentence_dataloader,device,valid_y,V_ind,adjustments):
    sentence_model.eval()
    sentence_preds = []
    sentence_probability=[]
    print(adjustments)
    scores=[]
    p=0
    for batch in valid_sentence_dataloader:
        idx,encoded_texts = batch
        encoded_texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in encoded_texts.items()}
        with torch.no_grad():
            outputs = sentence_model(**encoded_texts)
            sentence_probability.extend(nn.Softmax(dim=1)(outputs.logits-adjustments)[:,1].cpu().numpy().tolist())
            if p==0:
                print(nn.Softmax(dim=1)(outputs.logits))
                print(nn.Softmax(dim=1)(outputs.logits-adjustments))
                p=1
    for i in range(len(sentence_probability)):
        if sentence_probability[i]<=threshold:
            sentence_preds.append(0)
        else:
            sentence_preds.append(1)

    cur_ind=0
    curlabel=1
    pred_y=[]
    for _ in range(len(V_ind)):
        if cur_ind==V_ind[_]:
            curlabel=min(curlabel,sentence_preds[_])
        else:
            pred_y.append(curlabel)
            cur_ind+=1
            curlabel=sentence_preds[_]

    pred_y.append(curlabel)
    instance_acc = accuracy_score(valid_y, pred_y)
    instance_recall = recall_score(valid_y, pred_y)
    instance_neg_recall = recall_score(valid_y, pred_y, pos_label=0)
    instance_avg_acc=(instance_recall+instance_neg_recall)/2
    instance_precision = precision_score(valid_y, pred_y)
    instance_f1 = f1_score(valid_y, pred_y)
    instance_confusion_matrix=confusion_matrix(valid_y, pred_y)
    instance_classification_report=classification_report(valid_y, pred_y)
    print('instance_acc')
    print(instance_acc)
    print('instance_recall')
    print(instance_recall)
    print('instance_neg_recall')
    print(instance_neg_recall)
    print('instance_avg_acc')
    print(instance_avg_acc)
    print('instance_precision')
    print(instance_precision)
    print('instance_f1')
    print(instance_f1)
    print('instance_confusion_matrix')
    print(instance_confusion_matrix)
    print('instance_classification_report')
    print(instance_classification_report)
    args_dict['instance_acc']=instance_acc
    args_dict['instance_recall']=instance_recall
    args_dict['instance_neg_recall']=instance_neg_recall
    args_dict['instance_avg_acc']=instance_avg_acc
    args_dict['instance_precision']=instance_precision
    args_dict['instance_f1']=instance_f1

def valid_bag(model,valid_dataloader,device):
    model.eval()
    valid_preds = []
    true_labels = []
    for batch in valid_dataloader:
        idx,encoded_texts, labels = batch
        encoded_texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in encoded_texts.items()}
        with torch.no_grad():
            outputs = model(**encoded_texts)
            valid_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist())
            true_labels.extend(labels.numpy().tolist())
    bag_acc = accuracy_score(true_labels, valid_preds)
    bag_recall = recall_score(true_labels, valid_preds)
    bag_neg_recall = recall_score(true_labels, valid_preds,pos_label=0)
    bag_avg_acc=(bag_recall+bag_neg_recall)/2
    bag_precision = precision_score(true_labels, valid_preds)
    bag_f1 = f1_score(true_labels, valid_preds)
    bag_confusion_matrix=confusion_matrix(true_labels, valid_preds)
    bag_classification_report=classification_report(true_labels, valid_preds)
    print('bag_acc')
    print(bag_acc)
    print('bag_recall')
    print(bag_recall)
    print('bag_neg_recall')
    print(bag_neg_recall)
    print('bag_avg_acc')
    print(bag_avg_acc)
    print('bag_precision')
    print(bag_precision)
    print('bag_f1')
    print(bag_f1)
    print('bag_confusion_matrix')
    print(bag_confusion_matrix)
    print('bag_classification_report')
    print(bag_classification_report)
    args_dict['bag_acc']=bag_acc
    args_dict['bag_recall']=bag_recall
    args_dict['bag_neg_recall']=bag_neg_recall
    args_dict['bag_avg_acc']=bag_avg_acc
    args_dict['bag_precision']=bag_precision
    args_dict['bag_f1']=bag_f1

bag_acc=[]
bag_auc=[]
bag_recall=[]
bag_neg_recall=[]
bag_avg_acc=[]
bag_precision=[]
bag_f1=[]

avg_acc=[]
avg_auc=[]
avg_recall=[]
avg_neg_recall=[]
avg_avg_acc=[]
avg_precision=[]
avg_f1=[]

std_acc=[]
std_auc=[]
std_recall=[]
std_neg_recall=[]
std_avg_acc=[]
std_precision=[]
std_f1=[]

seeds=[0,1,2]
if args.dataset in ['imdb','amazon','yelp']:
    list_instance_imbalance=[int(i ) for i in [2,3,4,5] ]# 5
else:
    list_instance_imbalance=[int(i ) for i in [2,4,6,8,10] ]


for instance_imbalance in list_instance_imbalance:
    num_train_negative=len(positive_train_texts)//(bag_imbalance*(max_instance_imbalance+1)+max_instance_imbalance)
    num_train_positive=num_train_negative*(bag_imbalance*(instance_imbalance+1)+instance_imbalance)
    num_test_negative=len(positive_test_texts)//(instance_imbalance*2+1)
    num_test_positive=num_test_negative*(instance_imbalance*2+1)
    
    _bag_acc=[]
    _bag_recall=[]
    _bag_neg_recall=[]
    _bag_avg_acc=[]
    _bag_auc=[]
    _bag_precision=[]
    _bag_f1=[]
    
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True   

        negative_train_sentence=random.sample(negative_train_texts,num_train_negative)
        positive_train_sentence=random.sample(positive_train_texts,num_train_positive)
        negative_test_sentence=random.sample(negative_test_texts,num_test_negative)
        positive_test_sentence=random.sample(positive_test_texts,num_test_positive)

        positive_train_bags=[]
        negative_train_bags=[]
        positive_test_bags=[]
        negative_test_bags=[]

        positive_train_instance_labels=[]
        negative_train_instance_labels=[]
        positive_test_instance_labels=[]
        negative_test_instance_labels=[]

        for i in range(num_train_negative):
            pos_sen=positive_train_sentence[i*instance_imbalance:(i+1)*instance_imbalance]
            random_number=random.randint(0,instance_imbalance)
            bag=pos_sen[:random_number]+[negative_train_sentence[i]]+pos_sen[random_number:]
            instance_labels=[1 for i in range(random_number)]+[0]+[1 for i in range(len(pos_sen)-random_number)]
            negative_train_bags.append(bag)
            negative_train_instance_labels.append(instance_labels)

        for i in range(num_train_negative*bag_imbalance):
            pos_sen=positive_train_sentence[num_train_negative*instance_imbalance+i*(instance_imbalance+1):num_train_negative*instance_imbalance+(i+1)*(instance_imbalance+1)]
            bag=pos_sen
            instance_labels=[1 for i in range(len(pos_sen))]
            positive_train_bags.append(bag)
            positive_train_instance_labels.append(instance_labels)

        for i in range(num_test_negative):
            pos_sen=positive_test_sentence[i*instance_imbalance:(i+1)*instance_imbalance]
            random_number=random.randint(0,instance_imbalance)
            bag=pos_sen[:random_number]+[negative_test_sentence[i]]+pos_sen[random_number:]
            instance_labels=[1 for i in range(random_number)]+[0]+[1 for i in range(len(pos_sen)-random_number)]
            negative_test_bags.append(bag)
            negative_test_instance_labels.append(instance_labels)

        for i in range(num_test_negative):
            pos_sen=positive_test_sentence[num_test_negative*instance_imbalance+i*(instance_imbalance+1):num_test_negative*instance_imbalance+(i+1)*(instance_imbalance+1)]
            bag=pos_sen
            instance_labels=[1 for i in range(len(pos_sen))]
            positive_test_bags.append(bag)
            positive_test_instance_labels.append(instance_labels)

        print('len_positive_test_bags')
        print(len(positive_test_bags))
        print('len_negative_test_bags')
        print(len(negative_test_bags))
        print('len_positive_train_bags')
        print(len(positive_train_bags))
        print('len_negative_train_bags')
        print(len(negative_train_bags))

        test_y=[0 for _ in range(len(negative_test_bags))]+[1 for _ in range(len(positive_test_bags))]

        P_sentences=[]
        P_sentences_labels=[]
        U_sentences=[]
        U_sentences_labels=[]
        T_sentences=[]
        T_sentences_labels=[]
        ind_P=[]
        ind_U=[]
        ind_T=[]
        rev_ind_U=[]

        cur_i=0
        for _ in range(len(negative_train_bags)):
            U_sentences.extend(negative_train_bags[_])
            ind_U.extend([ _  for i in range(len(negative_train_bags[_]))])
            rev_ind_U.append([cur_i+i for i in range(len(negative_train_bags[_]))])
            cur_i+=len(negative_train_bags[_])

        
        for _ in range(len(positive_train_bags)):
            P_sentences.extend(positive_train_bags[_])
            ind_P.extend([ _ + len(negative_train_bags) for i in range(len(positive_train_bags[_]))])
            P_sentences_labels.extend(positive_train_instance_labels[_])
 
        ind_train=ind_U+ind_P
        test_bags=negative_test_bags+positive_test_bags
        test_instance_labels=negative_test_instance_labels+positive_test_instance_labels
        for _ in range(len(test_bags)):
            T_sentences.extend(test_bags[_])
            ind_T.extend([_ for i in range(len(test_bags[_]))])

        max_len_sentence=0
        sum_len_sentence=0
        num_sentence=0
    
        train_sentences=U_sentences+P_sentences
        train_PU_y=[0 for _ in range(len(U_sentences))]+[1 for _ in range(len(P_sentences))]
     
        sentence_tokenizer = RobertaTokenizer.from_pretrained("roberta-base", local_files_only=True)
        if args.pretrain:
            model_dir="./roberta-model"+'_'+args.dataset+'_'+'sentence'+'_'+str(bag_imbalance)+'_'+str(instance_imbalance)+'_'+str(seed)
            log_dir="./logs"+'_'+args.dataset+'_'+'sentence'+'_'+str(bag_imbalance)+'_'+str(instance_imbalance)+'_'+str(seed)
            text_dir="./pretrain_texts/"+args.dataset+'_'+'sentence'+'_'+str(bag_imbalance)+'_'+str(instance_imbalance)+'_'+str(seed)+'.txt'
            with open(text_dir, "w", encoding="utf-8") as file:
                for text in train_sentences:
                    file.write(text + "\n")
            if not os.path.exists(model_dir):
                pretrained_model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)
                data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
                pretrain_dataset=LineByLineTextDataset(tokenizer=tokenizer, file_path=text_dir,block_size=args.max_len)
                training_args = TrainingArguments(
                    output_dir=model_dir,
                    num_train_epochs=epoch, 
                    per_device_train_batch_size=batch_size,
                    logging_dir=log_dir,
                )
                trainer = Trainer(
                    model=pretrained_model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=pretrain_dataset
                )
                trainer.train()
                trainer.save_model(model_dir)
            sentence_model = RobertaForSequenceClassification.from_pretrained(model_dir,output_hidden_states=True, local_files_only=True).to(device)
        else:
            sentence_model = RobertaForSequenceClassification.from_pretrained("roberta-base",output_hidden_states=True, local_files_only=True).to(device)

        dataset_U=TextDataset(U_sentences,sentence_tokenizer,max_length=args.max_len)

        labeled_dataset_PU=LabeledTextDataset(train_sentences, train_PU_y, sentence_tokenizer, max_length=args.max_len)

        dataset_T=TextDataset(T_sentences, sentence_tokenizer, max_length=args.max_len)

        sampler_U=SequentialSampler(dataset_U)

        labeled_sampler_PU=RandomSampler(labeled_dataset_PU)

        sampler_T=SequentialSampler(dataset_T)

        dataloader_U=DataLoader(dataset_U,batch_size=batch_size,sampler=sampler_U)
        labeled_dataloader_PU=DataLoader(labeled_dataset_PU,batch_size=batch_size,sampler=labeled_sampler_PU)
        dataloader_T=DataLoader(dataset_T,batch_size=batch_size,sampler=sampler_T)

        if weight_decay is not None:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = []
            for name, param in sentence_model.named_parameters():
                optimizer_grouped_parameters.append({'params': [param], 'weight_decay': weight_decay if not any(nd in name for nd in no_decay) else 0.0})
            sentence_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        else:
            optimizer_grouped_parameters=model.named_parameters()
            sentence_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
        sentence_scheduler = lr_scheduler.CosineAnnealingLR(sentence_optimizer,T_max=epoch*math.ceil(len(train_sentences)/batch_size)+epoch*math.ceil(len(rev_ind_U)*2/batch_size))

        prior=instance_imbalance/(instance_imbalance+1)
        for i in range(epoch):
            sentence_model.train()
            epoch_pslab = torch.zeros(len(labeled_dataset_PU), 2).to(device)

            for idx,texts, labels in labeled_dataloader_PU:
                texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                labels = labels.to(device)
                outputs = sentence_model(**texts)
                logits=outputs.logits
                unlabeled,positive =  labels==0,labels==1
                n_unlabeled =labels[unlabeled].shape[0]
                n_positive=labels[positive].shape[0]
                
                loss=0
                epoch_pslab[idx] = logits.clone().detach()/torch.mean(logits,dim=0)
                if n_unlabeled is not  0:
                    weight=nn.Softmax()(logits[unlabeled,0]).detach()
                    loss_1 = nn.CrossEntropyLoss(reduction='none')(logits[unlabeled], labels[unlabeled])
                    loss_1=torch.sum(weight*loss_1)
                    loss+=loss_1
                if n_positive is not 0:
                    weight=nn.Softmax()(logits[positive,0]).detach()
                    loss_2 = nn.CrossEntropyLoss(reduction='none')(logits[positive], labels[positive])
                    loss_2=torch.sum(weight*loss_2)
                    loss+=loss_2

                loss=loss/prior

                sentence_model.zero_grad()
                loss.backward()

                sentence_optimizer.step()
                if args.scheduler:
                    sentence_scheduler.step()
            if use_max:
                idxs=[]
                for _ in range(len(rev_ind_U)):
                    indexs=rev_ind_U[_]
                    max_value,max_index=torch.max(epoch_pslab[indexs,0],dim=0)
                    idxs.append(indexs[max_index])

                max_dataset=LabeledTextDataset([train_sentences[_idx] for _idx in idxs], [0 for _ in range(len(idxs))], sentence_tokenizer, max_length=args.max_len)
                max_sampler=RandomSampler(max_dataset)
                max_dataloader=DataLoader(max_dataset,batch_size=batch_size,sampler=max_sampler)

                for idx,texts, labels in max_dataloader:
                    texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                    labels = labels.to(device)
                    outputs = sentence_model(**texts)
                    logits=outputs.logits
                    loss_passage=nn.CrossEntropyLoss()(logits,labels)
                    sentence_model.zero_grad()
                    loss_passage.backward()
                    sentence_optimizer.step()
                    
            if use_hard:
                hard_labels=torch.ones(len(labeled_dataset_PU)).to(device)
                for _ in range(len(rev_ind_U)):
                    indexs=rev_ind_U[_]
                    max_value,max_index=torch.max(epoch_pslab[indexs,0],dim=0)
                    hard_labels[indexs[max_index]]=0
                hard_labels = hard_labels.long()

                for idx,texts, labels in labeled_dataloader_PU:
                    texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                    h_labels=hard_labels[idx]
                    outputs = sentence_model(**texts)
                    logits=outputs.logits
                    negative ,positive =  h_labels==0,h_labels==1
                    positive_risk = prior*nn.CrossEntropyLoss()(logits[positive], h_labels[positive])
                    negative_risk = nn.CrossEntropyLoss()(logits[negative], h_labels[negative])
                    loss_passage=positive_risk+negative_risk
                    sentence_model.zero_grad()
                    loss_passage.backward()
                    sentence_optimizer.step()

            if use_balance:
                idxs_max=[]
                idxs_min=[]
                for _ in range(len(rev_ind_U)):
                    indexs=rev_ind_U[_]
                    max_value,max_index=torch.max(epoch_pslab[indexs,0],dim=0)
                    min_value,min_index=torch.min(epoch_pslab[indexs,0],dim=0)
                    idxs_max.append(indexs[max_index])
                    idxs_min.append(indexs[min_index])
                max_min_dataset=LabeledTextDataset([train_sentences[_idx] for _idx in idxs_max+idxs_min], [0 for _ in range(len(idxs_max))]+[1 for _ in range(len(idxs_min))], sentence_tokenizer, max_length=args.max_len)
                max_min_sampler=RandomSampler(max_min_dataset)
                max_min_dataloader=DataLoader(max_min_dataset,batch_size=batch_size,sampler=max_min_sampler)
                for idx,texts, labels in max_min_dataloader:
                    texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                    labels = labels.to(device)
                    outputs = sentence_model(**texts)
                    logits=outputs.logits
                    unlabeled,positive  = labels==0, labels==1
                    n_positive=labels[positive].shape[0]
                    loss_passage=nn.CrossEntropyLoss()(logits,labels)
                    loss_passage=loss_passage #/prior
                    sentence_model.zero_grad()
                    loss_passage.backward()
                    sentence_optimizer.step()
                    if args.scheduler:
                        sentence_scheduler.step()
                
        sentence_model.eval()

        pslab = torch.zeros(len(dataset_U), 2).to(device)

        for idx,texts in dataloader_U:
            texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
            outputs = sentence_model(**texts)
            logits=outputs.logits
            pslab[idx] = logits.clone().detach()
        
        sorted_probability = sorted(nn.Softmax(dim=1)(pslab)[:,1])
        threshold=sorted_probability[len(rev_ind_U)]
        print('instance_imbalance',file=text_file)
        print(instance_imbalance,file=text_file)
        print('seed',file=text_file)
        print(seed,file=text_file)
        print('valid_validation',file=text_file)
        valid_instance_threshold(sentence_model,dataloader_T,device,test_y,ind_T,threshold)
        _bag_acc.append(args_dict['instance_acc'])
        _bag_recall.append(args_dict['instance_recall'])
        _bag_neg_recall.append(args_dict['instance_neg_recall'])
        _bag_avg_acc.append(args_dict['instance_avg_acc'])
        _bag_precision.append(args_dict['instance_precision'])
        _bag_f1.append(args_dict['instance_f1'])
        _bag_auc.append(args_dict['instance_auc'])
        d = {}
        d['instance_imbalance'] = instance_imbalance
        d['seed'] = seed
        d['acc'] = args_dict['instance_acc']
        d['recall'] = args_dict['instance_recall']
        d['neg_recall'] = args_dict['instance_neg_recall']
        d['avg_acc'] = args_dict['instance_avg_acc']
        d['precision'] = args_dict['instance_precision']
        d['f1'] = args_dict['instance_f1']
        d['auc'] = args_dict['instance_auc']
        print(d,file=text_file)
        r.writerow(d)
        f.flush()
    bag_acc.append(_bag_acc)
    bag_recall.append(_bag_recall)
    bag_neg_recall.append(_bag_neg_recall)
    bag_avg_acc.append(_bag_avg_acc)
    bag_precision.append(_bag_precision)
    bag_f1.append(_bag_f1)
    bag_auc.append(_bag_auc)
    print('instance_imbalance',file=text_file)
    print(instance_imbalance,file=text_file)

    avg_acc.append(np.mean(_bag_acc))
    avg_recall.append(np.mean(_bag_recall))
    avg_neg_recall.append(np.mean(_bag_neg_recall))
    avg_avg_acc.append(np.mean(_bag_avg_acc))
    avg_precision.append(np.mean(_bag_precision))
    avg_f1.append(np.mean(_bag_f1))
    avg_auc.append(np.mean(_bag_auc))
    std_acc.append(np.std(_bag_acc))
    std_recall.append(np.std(_bag_recall))
    std_neg_recall.append(np.std(_bag_neg_recall))
    std_avg_acc.append(np.std(_bag_avg_acc))
    std_precision.append(np.std(_bag_precision))
    std_f1.append(np.std(_bag_f1))
    std_auc.append(np.std(_bag_auc))
    print('avg_acc',file=text_file)
    print(np.mean(_bag_acc),file=text_file)
    print('std_acc',file=text_file)
    print(np.std(_bag_acc),file=text_file)
    print('avg_recall',file=text_file)
    print(np.mean(_bag_recall),file=text_file)
    print('std_recall',file=text_file)
    print(np.std(_bag_recall),file=text_file)
    print('avg_neg_recall',file=text_file)
    print(np.mean(_bag_neg_recall),file=text_file)
    print('std_neg_recall',file=text_file)
    print(np.std(_bag_neg_recall),file=text_file)
    print('avg_avg_acc',file=text_file)
    print(np.mean(_bag_avg_acc),file=text_file)
    print('std_avg_acc',file=text_file)
    print(np.std(_bag_avg_acc),file=text_file)
    print('avg_precision',file=text_file)
    print(np.mean(_bag_precision),file=text_file)
    print('std_precision',file=text_file)
    print(np.std(_bag_precision),file=text_file)
    print('avg_f1',file=text_file)
    print(np.mean(_bag_f1),file=text_file)
    print('std_f1',file=text_file)
    print(np.std(_bag_f1),file=text_file)
    print('avg_auc',file=text_file)
    print(np.mean(_bag_auc),file=text_file)
    print('std_auc',file=text_file)
    print(np.std(_bag_auc),file=text_file)
    f.flush()
print('avg_acc',file=text_file)
print(avg_acc,file=text_file)
print('avg_recall',file=text_file)
print(avg_recall,file=text_file)
print('avg_neg_recall',file=text_file)
print(avg_neg_recall,file=text_file)
print('avg_avg_acc',file=text_file)
print(avg_avg_acc,file=text_file)
print('avg_precision',file=text_file)
print(avg_precision,file=text_file)
print('avg_f1',file=text_file)
print(avg_f1,file=text_file)
print('avg_auc',file=text_file)
print(avg_auc,file=text_file)
print('std_acc',file=text_file)
print(std_acc,file=text_file)
print('std_recall',file=text_file)
print(std_recall,file=text_file)
print('std_neg_recall',file=text_file)
print(std_neg_recall,file=text_file)
print('std_avg_acc',file=text_file)
print(std_avg_acc,file=text_file)
print('std_precision',file=text_file)
print(std_precision,file=text_file)
print('std_f1',file=text_file)
print(std_f1,file=text_file)
print('std_auc',file=text_file)
print(std_auc,file=text_file)
f.flush()

wandb.finish()
