import pickle
import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling,LineByLineTextDataset
from transformers import RobertaTokenizer, RobertaForMaskedLM,RobertaForSequenceClassification
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from datasets import load_dataset
import faiss
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
#import nltk
import faiss
#from nltk.corpus import wordnet
from googletrans import Translator
from googletrans import LANGUAGES
import wandb
import yaml
import csv
import jieba
# from synonyms import synonyms
import random
import copy
import numpy as np
import math
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import gensim.downloader as downloader
# embedding = downloader.load("word2vec-google-news-300")
import random
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
    # config={
    # "random_seed": 5,
    # "num_valid_negative": 100,
    # "imbalance_algorithm": "logit_adjustment",
    # "imbalance_ratio": 100,
    # "base_model":"base_model",
    # "weight_decay":0,
    # "valid_iteration":1000,
    # "epoch":20,
    # "lr":1e-5,
    # "batch_size":16,
    # "tro":2.0,
    # "scheduler":True,
    # "augmentation":"synonyms",
    # "synonyms":5,
    # "model_path":"seed5.pth"
    # }
)
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
# parser.add_argument('--imbalance_algorithm', type=str, default='logit_adjustment')
# parser.add_argument('--imbalance_ratio', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--used', type=float, default=0.9)
parser.add_argument('--bag_imbalance', type=int, default=1)
parser.add_argument('--max_instance_imbalance', type=int, default=10)
parser.add_argument('--self_epoch', type=int, default=10)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--self_sentence_epoch', type=int, default=10)
parser.add_argument('--ood_epoch', type=int, default=1)
parser.add_argument('--lambda_ood', type=float, default=1)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--dataset', type=str, default='sst2')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--tro', type=float, default=0.1)
parser.add_argument('--scheduler', type=bool, default=True)
parser.add_argument('--synonyms', type=int, default=1)
# parser.add_argument('--lambda_ood', type=float, default=1)
# parser.add_argument('--augmentation', type=str, default='synonyms')
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--augmentation', type=str, default=None)
parser.add_argument('--model', type=str, default='roberta-base')
parser.add_argument('--self_model_path', type=str, default='self_model')
parser.add_argument('--model_path', type=str, default='model')
parser.add_argument('--sentence_self_model_path', type=str, default='sentence_self_model')
parser.add_argument('--sentence_model_path', type=str, default='sentence_model')
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--tgt_lang',type=str,default='fr')
parser.add_argument('--use_max',type=bool,default=False)
parser.add_argument('--use_balance',type=bool,default=False)
parser.add_argument('--use_hard',type=bool,default=False)
parser.add_argument('--temperature',type=float,default=1.0)
parser.add_argument('--eta',type=float,default=5.0)
parser.add_argument('--self_passage', type=bool, default=False)
parser.add_argument('--self_sentence', type=bool, default=False)
parser.add_argument('--load_self_passage', type=bool, default=False)
parser.add_argument('--load_self_sentence', type=bool, default=False)
parser.add_argument('--train_passage', type=bool, default=False)
parser.add_argument('--train_sentence', type=bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--log_file', type=str, default='')
parser.add_argument('--hint', type=str, default='')
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
    # df = pd.read_csv('./data/sst2/dev.tsv', sep='\t')
    # valid_labels = df['label'].tolist()
    # valid_texts = df['content'].tolist()
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

    positive_train_texts=positive_texts[:int(len(positive_texts)*0.8)]
    negative_train_texts=negative_texts[:int(len(negative_texts)*0.8)]
    positive_train_labels=positive_labels[:int(len(positive_labels)*0.8)]
    negative_train_labels=negative_labels[:int(len(negative_labels)*0.8)]

    positive_test_texts=positive_texts[int(len(positive_texts)*0.8):]
    negative_test_texts=negative_texts[int(len(negative_texts)*0.8):]
    positive_test_labels=positive_labels[int(len(positive_labels)*0.8):]
    negative_test_labels=negative_labels[int(len(negative_labels)*0.8):]
elif args.dataset=='sms':
    dataset = load_dataset('sms_spam')#,data_dir='/data/jialh/sms')
    train_dataset,test_dataset = dataset['train'].train_test_split(test_size=0.2).values()
    #print(dataset)
    #test_dataset = dataset['test']
    positive_train_texts, positive_train_labels  = [train_dataset[i]['sms'] for i in range(len(train_dataset)) if train_dataset[i]['label']==1],[train_dataset[i]['label'] for i in range(len(train_dataset)) if train_dataset[i]['label']==1]
    negative_train_texts, negative_train_labels  = [train_dataset[i]['sms'] for i in range(len(train_dataset)) if train_dataset[i]['label']==0],[train_dataset[i]['label'] for i in range(len(train_dataset)) if train_dataset[i]['label']==0]
    positive_test_texts, positive_test_labels  = [test_dataset[i]['sms'] for i in range(len(test_dataset)) if test_dataset[i]['label']==1],[test_dataset[i]['label'] for i in range(len(test_dataset)) if test_dataset[i]['label']==1]
    negative_test_texts, negative_test_labels  = [test_dataset[i]['sms'] for i in range(len(test_dataset)) if test_dataset[i]['label']==0],[test_dataset[i]['label'] for i in range(len(test_dataset)) if test_dataset[i]['label']==0]

elif args.dataset=='toxic':
    dataset=load_dataset("jigsaw_toxicity_pred", data_dir='./toxic')
    #dataset = load_dataset('toxicity')#,data_dir='/data/jialh/sms')
    #train_dataset,test_dataset = dataset['train'].train_test_split(test_size=0.2).values()
    #print(dataset)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    print(train_dataset)
    positive_train_texts, positive_train_labels = [train_dataset[i]['comment_text'] for i in range(len(train_dataset)) if train_dataset[i]['toxic']==0], [1 for i in range(len(train_dataset)) if train_dataset[i]['toxic']==0]
    negative_train_texts, negative_train_labels = [train_dataset[i]['comment_text'] for i in range(len(train_dataset)) if train_dataset[i]['toxic']==1], [0 for i in range(len(train_dataset)) if train_dataset[i]['toxic']==1]
    positive_test_texts, positive_test_labels = [test_dataset[i]['comment_text'] for i in range(len(test_dataset)) if test_dataset[i]['toxic']==0], [1 for i in range(len(test_dataset)) if test_dataset[i]['toxic']==0]
    negative_test_texts, negative_test_labels = [test_dataset[i]['comment_text'] for i in range(len(test_dataset)) if test_dataset[i]['toxic']==1], [0 for i in range(len(test_dataset)) if test_dataset[i]['toxic']==1]
elif args.dataset=='yelp':
    dataset = load_dataset('yelp_polarity')
    train_texts,train_labels = dataset['train']['text'],dataset['train']['label']
    test_texts,test_labels = dataset['test']['text'],dataset['test']['label']
    positive_train_texts, positive_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==1],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==1]
    negative_train_texts, negative_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==0],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==0]
    positive_test_texts, positive_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==1],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==1]
    negative_test_texts, negative_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==0],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==0]


elif args.dataset=='imdb':
    dataset = load_dataset('imdb',data_dir='/home/jialh/.cache/huggingface/datasets/imdb')
    train_texts,train_labels = dataset['train']['text'],dataset['train']['label']
    test_texts,test_labels = dataset['test']['text'],dataset['test']['label']
    positive_train_texts, positive_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==1],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==1]
    negative_train_texts, negative_train_labels  = [train_texts[i] for i in range(len(train_texts)) if train_labels[i]==0],[train_labels[i] for i in range(len(train_texts)) if train_labels[i]==0]
    positive_test_texts, positive_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==1],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==1]
    negative_test_texts, negative_test_labels  = [test_texts[i] for i in range(len(test_texts)) if test_labels[i]==0],[test_labels[i] for i in range(len(test_texts)) if test_labels[i]==0]
    
elif args.dataset=='amazon':
    dataset = load_dataset('amazon_polarity',data_dir='/data/jialh/amazon_polarity')
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

elif args.dataset=='sentiment140':
    df = pd.read_csv('./data/sentiment140/training.1600000.processed.noemoticon.csv',encoding="ISO-8859-1")
    texts = df.iloc[:, 5]
    labels = df.iloc[:, 0]
    positive_texts = [texts[i] for i in range(len(texts)) if labels[i]==4]
    negative_texts = [texts[i] for i in range(len(texts)) if labels[i]==0]
    positive_labels = [1 for i in range(len(texts)) if labels[i]==4]
    negative_labels = [0 for i in range(len(texts)) if labels[i]==0]
    positive_train_texts=positive_texts[:int(len(positive_texts)*0.8)]
    negative_train_texts=negative_texts[:int(len(negative_texts)*0.8)]
    positive_train_labels=positive_labels[:int(len(positive_labels)*0.8)]
    negative_train_labels=negative_labels[:int(len(negative_labels)*0.8)]
    positive_test_texts=positive_texts[int(len(positive_texts)*0.8):]
    negative_test_texts=negative_texts[int(len(negative_texts)*0.8):]
    positive_test_labels=positive_labels[int(len(positive_labels)*0.8):]
    negative_test_labels=negative_labels[int(len(negative_labels)*0.8):]

device = args.device
bag_imbalance=args.bag_imbalance
max_instance_imbalance=args.max_instance_imbalance
model_name=args.model
epoch=args.epoch
batch_size = args.batch_size
weight_decay=args.weight_decay
tro=args.tro
lr=args.lr
used=args.used
use_max=args.use_max
use_balance=args.use_balance
use_hard=args.use_hard
eps=1e-6
eta=args.eta

class synonym_replacement:
    def __init__(self, num=5):
        self.num=num
    def augment(self, sentence):
        words = nltk.word_tokenize(sentence)
        # print(words)
        # print(len(words))
        # augmented_sentences = [sentence]
        # for _ in range(self.num):
        new_sentence = sentence
        num=min(self.num,len(words))
        random_index=random.sample(range(len(words)), num)
        for i in range(len(random_index)):
            rand_word_index = random_index[i]
            word=words[rand_word_index]
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
            synonyms = list(synonyms)
            if synonyms:
                synonym = random.choice(synonyms)
                new_sentence = new_sentence.replace(word, synonym)
        return new_sentence

class back_translate:
    def __init__(self,src_lang='en', tgt_lang='fr',device='cuda:0'):
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.device=device
        translate_model_name = f'Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}'
        self.translate_model = MarianMTModel.from_pretrained(translate_model_name).to(device)
        self.translate_tokenizer = MarianTokenizer.from_pretrained(translate_model_name)
        back_translate_model_name = f'Helsinki-NLP/opus-mt-{self.tgt_lang}-{self.src_lang}'
        self.back_translate_model = MarianMTModel.from_pretrained(back_translate_model_name).to(device)
        self.back_translate_tokenizer = MarianTokenizer.from_pretrained(back_translate_model_name)

    def augment(self,sentence):
        device=self.device
        inputs = self.translate_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
        translated_ids = self.translate_model.generate(**inputs)
        translated_text = self.translate_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
        inputs = self.back_translate_tokenizer(translated_text, return_tensors='pt', padding=True, truncation=True).to(device)
        back_translated_ids = self.back_translate_model.generate(**inputs)
        back_translated_text = self.back_translate_tokenizer.decode(back_translated_ids[0], skip_special_tokens=True)
        return back_translated_text

class PassageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,max_length=1024,augment=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.augment=augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text=self.texts[index]
        # text = ''.join(texts)
        # print('len')
        # print(len(text))
        label = self.labels[index]
        if self.augment:
            text=self.augment.augment(text)
            # text=self.augment.augment(text)
        # print('len')
        # print(len(text))
        encoded_text = self.tokenizer(text, padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt',return_token_type_ids=True)
        # print(encoded_text.keys())
        return index, encoded_text, label

class ContrastiveDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __getitem__(self, index):
        text = self.texts[index]
        text_aug_1=self.tokenizer(text,padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt', return_token_type_ids=True)
        text_aug_2=self.tokenizer(text,padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt', return_token_type_ids=True)
        return text_aug_1, text_aug_2

    def __len__(self): 
        return len(self.texts)

class RawDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

    def __len__(self): 
        return len(self.X)

class UDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        x = self.X[index]
        return x

    def __len__(self): 
        return len(self.X)

class TripleDataset(Dataset):
    def __init__(self, X, I, D):
        self.n = len(X)
        self.X = X
        self.I = I
        self.D = D
        self.pointer = [0 for i in range(X.shape[0])]
        self.random = 0
        self.K = self.D.shape[1]
        self.SEED = 1e9 + 7

    def __getitem__(self, index):
        self.pointer[index] += 1
        if self.pointer[index] >= self.K: self.pointer[index] = 1
        self.random = int((self.random + self.SEED) % self.n)

        xA = self.X[index, :]
        xB = self.X[self.I[index, self.pointer[index]], :]
        xC = self.X[self.random, :]

        return xA, xB, xC, self.D[index, self.pointer[index]]

    def __len__(self): 
        return len(self.X)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer,max_length=1024,augment=None,output_idx=True):
        self.texts = texts
        # self.labels = labels
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.augment=augment
        self.output_idx=output_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text=self.texts[index]
        # text = ''.join(texts)
        # label = self.labels[index]
        if self.augment:
            text=self.augment.augment(text)
        # print('len')
        # print(len(text))
        encoded_text = self.tokenizer(text, padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt',return_token_type_ids=True)
        # print(encoded_text.keys())
        if self.output_idx:
            return index, encoded_text
        else:
            return encoded_text

class LabeledTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer,max_length=1024,augment=None,output_idx=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length=max_length
        self.augment=augment
        self.output_idx=output_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text=self.texts[index]
        # text = ''.join(texts)
        label = self.labels[index]
        if self.augment:
            text=self.augment.augment(text)
        # print('len')
        # print(len(text))
        encoded_text = self.tokenizer(text, padding='max_length', 
            truncation=True, max_length=self.max_length, return_tensors='pt',return_token_type_ids=True)
        # print(encoded_text.keys())
        if self.output_idx:
            return index, encoded_text, label
        else:
            return encoded_text, label


 

def valid_instance(sentence_model,valid_sentence_dataloader,device,valid_y,V_ind,center,threshold):
    print(threshold)
    sentence_model.eval()
    sentence_preds = []
    scores=[]
    for batch in valid_sentence_dataloader:
        idx,encoded_texts = batch
        encoded_texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in encoded_texts.items()}
        with torch.no_grad():
            feature = sentence_model(**encoded_texts).hidden_states[-1][:,0,:]
            score=torch.sum((feature-center)**2,axis=1).detach().cpu().numpy()
            scores.append(score)
            #sentence_preds.extend(torch.argmax(outputs.hidden_states[-1][:,0,:], dim=1).cpu().numpy().tolist())
    scores=np.concatenate(scores,0)
    sentence_probability=1-scores/np.linalg.norm(scores)
    # print(scores)
    # print('mean')
    # print(np.means(scores))
    sentence_preds=(scores<=threshold).tolist()
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
    instance_confusion_matrix=confusion_matrix(valid_y, pred_y)
    instance_auc=roc_auc_score(valid_y, probs)
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
            # sentence_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist())
            sentence_probability.extend(nn.Softmax(dim=1)(outputs.logits)[:,1].cpu().numpy().tolist())
            # print(((nn.Sigmoid()(outputs.logits[0:]))>0.5).to(torch.int).squeeze())
            # sentence_preds.extend(((nn.Sigmoid()(outputs.logits))>0.5).to(torch.int).squeeze().cpu().numpy().tolist())
    indexed_list = list(enumerate(sentence_probability))
    sorted_probability = sorted(indexed_list, key=lambda x: x[1])
    # sorted_keys=[x[0]for x in  sorted_probability]
    # negative_keys=sorted_keys[:len(valid_y)//2]
    # print(threshold)
    threshold=[x[1] for x in sorted_probability][len(valid_y)//2]
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
    wandb.log({"instance_acc": instance_acc, "instance_recall": instance_recall, "instance_neg_recall":instance_neg_recall,"instance_avg_acc":instance_avg_acc,
        "instance_precision":instance_precision, "instance_f1":instance_f1,"instance_confusion_matrix":instance_confusion_matrix,
        "instance_classification_report":instance_classification_report})

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
            sentence_probability.extend(nn.Softplus()(outputs.logits.squeeze()).cpu().numpy().tolist())
    print(threshold)
    for i in range(len(sentence_probability)):
        if sentence_probability[i]<=threshold:
            sentence_preds.append(1)
        else:
            sentence_preds.append(0)

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
    wandb.log({"instance_acc": instance_acc, "instance_recall": instance_recall, "instance_neg_recall":instance_neg_recall,"instance_avg_acc":instance_avg_acc,
        "instance_precision":instance_precision, "instance_f1":instance_f1,"instance_confusion_matrix":instance_confusion_matrix,
        "instance_classification_report":instance_classification_report})

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
            # sentence_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy().tolist())
            sentence_probability.extend(nn.Softmax(dim=1)(outputs.logits-adjustments)[:,1].cpu().numpy().tolist())
            if p==0:
                print(nn.Softmax(dim=1)(outputs.logits))
                print(nn.Softmax(dim=1)(outputs.logits-adjustments))
                p=1
    threshold=0.5
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
    wandb.log({"instance_acc": instance_acc, "instance_recall": instance_recall, "instance_neg_recall":instance_neg_recall,"instance_avg_acc":instance_avg_acc,
        "instance_precision":instance_precision, "instance_f1":instance_f1,"instance_confusion_matrix":instance_confusion_matrix,
        "instance_classification_report":instance_classification_report})

def valid_bag(model,valid_dataloader,device):
    model.eval()
    valid_preds = []
    true_labels = []
    for batch in valid_dataloader:
        idx,encoded_texts, labels = batch
        encoded_texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in encoded_texts.items()}
        with torch.no_grad():
            outputs = model(**encoded_texts)
            # print(((nn.Sigmoid()(outputs.logits[0:]))>0.5).to(torch.int).squeeze())
            # print(nn.Sigmoid()(outputs.logits))
            # valid_preds.extend(((nn.Sigmoid()(outputs.logits))>0.5).to(torch.int).squeeze().cpu().numpy().tolist())
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
    wandb.log({"bag_acc": bag_acc, "bag_recall": bag_recall, "bag_neg_recall":bag_neg_recall,"bag_avg_acc":bag_avg_acc,
        "bag_precision":bag_precision, "bag_f1":bag_f1,"bag_confusion_matrix":bag_confusion_matrix,
        "bag_classification_report":bag_classification_report})

bag_acc=[]
bag_recall=[]
bag_neg_recall=[]
bag_avg_acc=[]
bag_precision=[]
bag_f1=[]
bag_auc=[]
avg_acc=[]
avg_recall=[]
avg_neg_recall=[]
avg_avg_acc=[]
avg_precision=[]
avg_f1=[]
avg_auc=[]
std_acc=[]
std_recall=[]
std_neg_recall=[]
std_avg_acc=[]
std_precision=[]
std_f1=[]
std_auc=[]
beta=args.beta
gamma=args.gamma

seeds=[0,1,2]
if args.dataset in ['imdb','amazon','yelp','toxic','sms']:
    list_instance_imbalance=[int(i ) for i in [2,3,4,5] ]# 5
else:
    list_instance_imbalance=[int(i ) for i in [8,10] ]

def get_radius(dist, nu):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

for instance_imbalance in list_instance_imbalance:
    num_train_negative=len(positive_train_texts)//(bag_imbalance*(max_instance_imbalance+1)+max_instance_imbalance)
    num_train_positive=num_train_negative*(bag_imbalance*(instance_imbalance+1)+instance_imbalance)
    num_test_negative=len(positive_test_texts)//(instance_imbalance*2+1)
    num_test_positive=num_test_negative*(instance_imbalance*2+1)
    
    _bag_acc=[]
    _bag_recall=[]
    _bag_neg_recall=[]
    _bag_avg_acc=[]
    _bag_precision=[]
    _bag_f1=[]
    _bag_auc=[]
    
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        # positive_train_sentence=random.sample(positive_train_texts,num_train_positive)
        # negative_train_sentence=random.sample(negative_train_texts,num_train_negative)
        # positive_test_sentence=random.sample(positive_test_texts,num_test_positive)
        # negative_test_sentence=random.sample(negative_test_texts,num_test_negative)

        # positive_train_bags=[]
        # negative_train_bags=[]
        # positive_test_bags=[]
        # negative_test_bags=[]
        
        # positive_train_instance_labels=[]
        # negative_train_instance_labels=[]
        # positive_test_instance_labels=[]
        # negative_test_instance_labels=[]

        # for i in range(num_train_negative*bag_imbalance):
        #     pos_sen=positive_train_sentence[i*(instance_imbalance+1):(i+1)*(instance_imbalance+1)]
        #     bag=pos_sen
        #     instance_labels=[1 for i in range(len(pos_sen))]
        #     positive_train_bags.append(bag)
        #     positive_train_instance_labels.append(instance_labels)

        # for i in range(num_train_negative):
        #     pos_sen=positive_train_sentence[num_train_negative*bag_imbalance*(instance_imbalance+1)+i*instance_imbalance:num_train_negative*bag_imbalance*(instance_imbalance+1)+(i+1)*instance_imbalance]
        #     random_number=random.randint(0,instance_imbalance)
        #     bag=pos_sen[:random_number]+[negative_train_sentence[i]]+pos_sen[random_number:]
        #     instance_labels=[1 for i in range(random_number)]+[0]+[1 for i in range(len(pos_sen)-random_number)]
        #     negative_train_bags.append(bag)
        #     negative_train_instance_labels.append(instance_labels)
        # for i in range(num_test_negative):
        #     pos_sen=positive_test_sentence[i*(instance_imbalance+1):(i+1)*(instance_imbalance+1)]
        #     bag=pos_sen
        #     instance_labels=[1 for i in range(len(pos_sen))]
        #     positive_test_bags.append(bag)
        #     positive_test_instance_labels.append(instance_labels)

        # for i in range(num_test_negative):
        #     pos_sen=positive_test_sentence[num_test_negative*(instance_imbalance+1)+i*instance_imbalance:num_test_negative*(instance_imbalance+1)+(i+1)*instance_imbalance]
        #     random_number=random.randint(0,instance_imbalance)
        #     bag=pos_sen[:random_number]+[negative_test_sentence[i]]+pos_sen[random_number:]
        #     instance_labels=[1 for i in range(random_number)]+[0]+[1 for i in range(len(pos_sen)-random_number)]
        #     negative_test_bags.append(bag)
        #     negative_test_instance_labels.append(instance_labels)
        # print('len_positive_test_bags')
        # print(len(positive_test_bags))
        # print('len_negative_test_bags')
        # print(len(negative_test_bags))
        # print('len_positive_train_bags')
        # print(len(positive_train_bags))
        # print('len_negative_train_bags')
        # print(len(negative_train_bags))

        # test_y=[1 for _ in range(len(positive_test_bags))]+[1 for _ in range(len(negative_test_bags))]
        # P_sentences=[]
        # P_sentences_labels=[]
        # U_sentences=[]
        # U_sentences_labels=[]
        # T_sentences=[]
        # T_sentences_labels=[]
        # ind_P=[]
        # ind_U=[]
        # ind_T=[]
        # rev_ind_U=[]

        # for _ in range(len(positive_train_bags)):
        #     P_sentences.extend(positive_train_bags[_])
        #     ind_P.extend([ _ for i in range(len(positive_train_bags[_]))])
        #     P_sentences_labels.extend(positive_train_instance_labels[_])

        # cur_i=len(P_sentences_labels)
        # for _ in range(len(negative_train_bags)):
        #     U_sentences.extend(negative_train_bags[_])
        #     ind_U.extend([ _ + len(positive_train_bags) for i in range(len(negative_train_bags[_]))])
        #     rev_ind_U.append([cur_i+i for i in range(len(negative_train_bags[_]))])
        #     cur_i+=len(negative_train_bags[_])
        #     U_sentences_labels.extend(negative_train_instance_labels[_])
        # ind_train=ind_P+ind_U
        # test_bags=positive_test_bags+negative_test_bags
        # test_instance_labels=positive_test_instance_labels+negative_test_instance_labels
        # for _ in range(len(test_bags)):
        #     T_sentences.extend(test_bags[_])
        #     ind_T.extend([_ for i in range(len(test_bags[_]))])
        #     T_sentences_labels.extend(test_instance_labels[_])
        # train_sentences=P_sentences+U_sentences
        # train_PU_y=[1 for _ in range(len(P_sentences))]+[0 for _ in range(len(U_sentences))]

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
            # T_sentences_labels.extend(test_instance_labels[_])

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
            sentence_model = RobertaForSequenceClassification.from_pretrained(model_dir,output_hidden_states=True, local_files_only=True, num_labels=1).to(device)
        else:
            sentence_model = RobertaForSequenceClassification.from_pretrained("roberta-base",output_hidden_states=True, local_files_only=True, num_labels=1).to(device)


        dataset_P=TextDataset(P_sentences,sentence_tokenizer,max_length=args.max_len)
        dataset_U=TextDataset(U_sentences,sentence_tokenizer,max_length=args.max_len)
        dataset_PU=TextDataset(train_sentences,sentence_tokenizer,max_length=args.max_len)

        labeled_dataset_PU=LabeledTextDataset(train_sentences, train_PU_y, sentence_tokenizer, max_length=args.max_len)
        # labeled_dataset_PN=LabeledTextDataset(train_sentences, train_PN_y, sentence_tokenizer, max_length=args.max_len)
        dataset_T=TextDataset(T_sentences, sentence_tokenizer, max_length=args.max_len)
        labeled_dataset_T=LabeledTextDataset(T_sentences, T_sentences_labels, sentence_tokenizer, max_length=args.max_len)
        
        sampler_P=SequentialSampler(dataset_P)
        sampler_U=SequentialSampler(dataset_U)
        sampler_PU=SequentialSampler(dataset_PU)
        labeled_sampler_T=RandomSampler(labeled_dataset_T)
        labeled_sampler_PU=RandomSampler(labeled_dataset_PU)
        # labeled_sampler_PN=RandomSampler(labeled_dataset_PN)
        sampler_T=SequentialSampler(dataset_T)

        dataloader_P=DataLoader(dataset_P,batch_size=batch_size,sampler=sampler_P)
        dataloader_U=DataLoader(dataset_U,batch_size=batch_size,sampler=sampler_U)
        dataloader_PU=DataLoader(dataset_PU,batch_size=batch_size,sampler=sampler_PU)
        labeled_dataloader_T=DataLoader(labeled_dataset_T,batch_size=batch_size,sampler=labeled_sampler_T)
        labeled_dataloader_PU=DataLoader(labeled_dataset_PU,batch_size=batch_size,sampler=labeled_sampler_PU)
        # labeled_dataloader_PN=DataLoader(labeled_dataset_PN,batch_size=batch_size,sampler=labeled_sampler_PN)
        dataloader_T=DataLoader(dataset_T,batch_size=batch_size,sampler=sampler_T)

        # dataset_contrastive=ContrastiveDataset(train_sentences,sentence_tokenizer,max_length=args.max_len)
        # sampler_contrastive=RandomSampler(dataset_contrastive)
        # dataloader_contrastive=DataLoader(dataset_contrastive,batch_size=batch_size,sampler=sampler_contrastive)
     
        # if weight_decay is not None:
        #     no_decay = ['bias', 'LayerNorm.weight']
        #     optimizer_grouped_parameters = []
        #     for name, param in sentence_model.named_parameters():
        #         optimizer_grouped_parameters.append({'params': [param], 'weight_decay': weight_decay if not any(nd in name for nd in no_decay) else 0.0})
        #     sentence_self_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        # else:
        #     optimizer_grouped_parameters=sentence_model.named_parameters()
        #     sentence_self_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

        # sentence_self_scheduler = lr_scheduler.CosineAnnealingLR(sentence_self_optimizer,T_max=math.ceil(args.self_sentence_epoch*len(train_sentences)/batch_size))

        # tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        # model = RobertaForSequenceClassification.from_pretrained("roberta-base",output_hidden_states=True).to(device)

        # labeled_dataset = PassageDataset(texts=train_x,  labels=train_y, tokenizer=tokenizer, max_length=args.max_len,augment=None)
        # labeled_sampler = RandomSampler(labeled_dataset)
        # labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size,sampler=labeled_sampler)

        # test_dataset = PassageDataset(texts=test_x, labels=test_y,tokenizer=tokenizer,max_length=args.max_len,augment=None)
        # test_sampler = SequentialSampler(test_dataset)
        # test_dataloader = DataLoader(test_dataset, batch_size=batch_size,sampler=test_sampler)
        
        if weight_decay is not None:
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = []
            for name, param in sentence_model.named_parameters():
                optimizer_grouped_parameters.append({'params': [param], 'weight_decay': weight_decay if not any(nd in name for nd in no_decay) else 0.0})
            sentence_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)
        else:
            optimizer_grouped_parameters=model.named_parameters()
            sentence_optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)

        sentence_scheduler = lr_scheduler.CosineAnnealingLR(sentence_optimizer,T_max=math.ceil(epoch*len(train_sentences)/batch_size))




        # for i in range(args.self_sentence_epoch):
        #     print(i)
        #     sentence_model.train()


        c = torch.zeros(sentence_model.config.hidden_size).to(device)
        sentence_model.eval()
        n_samples=0
        with torch.no_grad():
            for idx, texts, labels in labeled_dataloader_PU:
                texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                labels = labels.to(device)
                outputs = sentence_model(**texts).hidden_states[-1][:,0,:]
                n_samples+=outputs.shape[0]
                c+=torch.sum(outputs,dim=0)
        c/=n_samples
        epoch_dist = torch.zeros(len(labeled_dataset_PU)).to(device)
        R=torch.tensor(0,device=device)
        for i in range(epoch):
            sentence_model.train()

            for idx,texts, labels in labeled_dataloader_PU:
                texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                labels = labels.to(device)
                outputs = sentence_model(**texts).hidden_states[-1][:,0,:]
                dist = torch.sum((outputs - c) ** 2, dim=1)
                scores = dist - R ** 2
                loss = R ** 2 + (1 / 0.1) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                # loss = torch.mean(dist)
                losses = torch.where(labels == 1, ((dist + eps) ** 1), 0)
                loss = torch.mean(losses)
                sentence_model.zero_grad()
                loss.backward()
                sentence_optimizer.step()
                if args.scheduler:
                    sentence_scheduler.step()
                R.data = torch.tensor(get_radius(dist, 0.1), device=device)
            sentence_model.eval()

            for idx,texts, labels in labeled_dataloader_PU:
                texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = sentence_model(**texts).hidden_states[-1][:,0,:]
                    dist = torch.sum((outputs - c) ** 2, dim=1)
                    epoch_dist[idx]=dist
                               
            sentence_model.train()
            if use_max:
                idxs=[]
                for _ in range(len(rev_ind_U)):
                    indexs=rev_ind_U[_]
                    max_value,max_index=torch.max(epoch_dist[indexs],dim=0)
                    idxs.append(indexs[max_index])

                max_dataset=LabeledTextDataset([train_sentences[_idx] for _idx in idxs], [0 for _ in range(len(idxs))], sentence_tokenizer, max_length=args.max_len)
                max_sampler=RandomSampler(max_dataset)
                max_dataloader=DataLoader(max_dataset,batch_size=batch_size,sampler=max_sampler)

                for idx,texts, labels in max_dataloader:
                    texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                    labels = labels.to(device)
                    outputs = sentence_model(**texts).hidden_states[-1][:,0,:]
                    dist = torch.sum((outputs - c) ** 2, dim=1)
                    losses=((dist + eps) ** -1)*eta
                    loss_passage=torch.mean(losses)
                    sentence_model.zero_grad()
                    loss_passage.backward()
                    sentence_optimizer.step()

            if use_hard:
                hard_labels=torch.ones(len(labeled_dataset_PU)).to(device)
                for _ in range(len(rev_ind_U)):
                    indexs=rev_ind_U[_]
                    max_value,max_index=torch.max(epoch_dist[indexs],dim=0)
                    hard_labels[indexs[max_index]]=0
                hard_labels = hard_labels.long()

                for idx,texts, labels in labeled_dataloader_PU:
                    texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                    h_labels=hard_labels[idx]
                    outputs = sentence_model(**texts).hidden_states[-1][:,0,:]
                    dist = torch.sum((outputs - c) ** 2, dim=1)
                    losses = torch.where(h_labels == 1, eta * ((dist + eps) ** 1), eta * ((dist + eps) ** -1))
                    loss_passage=torch.mean(losses)
                    sentence_model.zero_grad()
                    loss_passage.backward()
                    sentence_optimizer.step()

            if use_balance:
                idxs_max=[]
                idxs_min=[]
                for _ in range(len(rev_ind_U)):
                    indexs=rev_ind_U[_]
                    max_value,max_index=torch.max(epoch_dist[indexs],dim=0)
                    min_value,min_index=torch.min(epoch_dist[indexs],dim=0)
                    idxs_max.append(indexs[max_index])
                    idxs_min.append(indexs[min_index])

                max_min_dataset=LabeledTextDataset([train_sentences[_idx] for _idx in idxs_max+idxs_min], [0 for _ in range(len(idxs_max))]+[1 for _ in range(len(idxs_min))], sentence_tokenizer, max_length=args.max_len)
                max_min_sampler=RandomSampler(max_min_dataset)
                max_min_dataloader=DataLoader(max_min_dataset,batch_size=batch_size,sampler=max_min_sampler)

                for idx,texts, labels in max_min_dataloader:
                    texts = {key: value.to(device).flatten(start_dim=0, end_dim=1) for key, value in texts.items()}
                    labels = labels.to(device)
                    outputs = sentence_model(**texts).hidden_states[-1][:,0,:]
                    dist = torch.sum((outputs - c) ** 2, dim=1)
                    losses = torch.where(labels == 1, eta * ((dist + eps) ** 1), eta * ((dist + eps) ** -1))
                    loss_passage=torch.mean(losses)
                    sentence_model.zero_grad()
                    loss_passage.backward()
                    sentence_optimizer.step()




        scores=epoch_dist.detach().cpu().numpy()
        sorted_probability = sorted(scores, reverse=True)
        threshold=sorted_probability[len(rev_ind_U)]
      
        print('instance_imbalance')
        print(instance_imbalance)
        print('seed')
        print(seed)
        print('valid_validation')
        valid_instance(sentence_model,dataloader_T,device,test_y,ind_T,c,threshold)
        # valid_bag(model,test_dataloader,device)
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

wandb.finish()
