# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:36:44 2018

@author: 广芮
"""
####################################################################remove tags


import re
from bs4 import BeautifulSoup as Soup
import re, nltk
from urllib.request import urlopen
import html2text
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
from distutils.filelist import findall
import codecs
from string import digits
import string
import time
from datetime import datetime
table = str.maketrans({key: None for key in string.punctuation})

remove_digits = str.maketrans('', '', digits)

def filter_tags(htmlstr):
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I)
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)
    re_br=re.compile('<br\s*?/?>')
    re_h=re.compile('</?\w+[^>]*>')
    re_comment=re.compile('<!--[^>]*-->')
    s=re_cdata.sub('',htmlstr)
    s=re_script.sub('',s) 
    s=re_style.sub('',s)
    s=re_br.sub('\n',s)
    s=re_h.sub('',s) 
    s=re_comment.sub('',s)
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    s=replaceCharEntity(s)
    return s

def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  
        key = sz.group('name')  
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


def repalce(s, re_exp, repl_string):
    return re_exp.sub(repl_string, s)

def Cleaning_data(x):
    m2=str(x).replace('<p>&nbsp; &nbsp; &nbsp; &nbsp;','').replace('</p><p><br></p>','').replace('<br>','').replace('</p>','').replace('<p>','').replace('       ','').strip()
    m3=filter_tags(m2)
    m4=replaceCharEntity(m3)
    return m4
####################################################################extract MDA
    
import os
import time
path="C:/Users/广芮/Desktop/capstone/test"
files=os.listdir(path)
Final_txt=[]
year=[]
Firm=[]
Failed_list=[]
final_list=[]
CIK=[]
DD=[]  


for f in files:
    add= 'file:///C:/Users/广芮/Desktop/capstone/test/'+f
    page=urlopen(add)
    contents = page.read() 
    soup=BeautifulSoup(contents,'lxml')  # read html as text
    text=soup.get_text()
    #clean txt
    clean_text=Cleaning_data(text)
    text_list = clean_text.split('.')
    clean_list=[txt.replace('\n','') for txt in text_list]
    clean_list=[txt.replace('\xa0','') for txt in clean_list]
    clean_list0=[txt.strip() for txt in clean_list]  
    
    total_str='.'.join(clean_list0)
    total_str = '.' + total_str
    # extract with regex
    a=re.compile(r'([^.]*?discussion[^.]*?and[^.]*?analysis[^.]*?of[^.]*?financial[^.]*?condition[^.]*?results[^.]*?operations[^.]*\.)', re.IGNORECASE)
    beg_target=a.findall(total_str)
    beg_target=[ori.replace('.','') for ori in beg_target]
    
    b=re.compile(r'([^.]*?and[^.]*?qualitative[^.]*?Disclosures[^.]*?about[^.]*?market[^.]*\.)', re.IGNORECASE)
    end_target1=b.findall(total_str)
    end_target1=[ori.replace('.','') for ori in end_target1]
    
    d=re.compile(r'([^.]*?FINANCIAL[^.]*?STATEMENTS[^.]*?and[^.]*?SUPPLEMENTARY[^.]*?data[^.]*\.)', re.IGNORECASE)
    end_target2=d.findall(total_str)
    end_target2=[ori.replace('.','') for ori in end_target2]  
        
    
    dd_1=re.compile(r'([^.]*?for[^.]*?the[^.]*?fiscal[^.]*?year[^.]*?ended[^.]*\.)', re.IGNORECASE)########################################################### 修改
    dd_target_1=dd_1.findall(total_str)[0]
    dd_target_1=dd_target_1.translate(table) 
    dd_target_1=dd_target_1.upper()
    dd_target_1=dd_target_1.replace(" ","")
    dd_target_1=dd_target_1.replace("\t","")
    date1=re.findall(r"ENDED[A-Z]{3,8}\d{6}",dd_target_1)[0][5:]
    d1=datetime.strptime(date1,"%B%d%Y").strftime("%Y-%m-%d")
    
    if len(d1)!=0:
        Date=d1
    else:
        dd_2=re.compile(r'([^.]*?for[^.]*?the[^.]*?transition[^.]*?period[^.]*?from[^.]*\.)', re.IGNORECASE)########################################################### 修改
        dd_target_2=dd_2.findall(total_str)
        dd_target2=dd_target_2[0]
        dd_target2=dd_target2.translate(table)
        dd_target2=dd_target2.upper()
        dd_target2.replace(" ","")
        dd_target2=dd_target2.replace("\t","")
        date2=re.findall(r"TO[A-Z]{3,8}\d{6}",dd_target2)[0][5:]
        d2=datetime.strptime(date2,"%B%d%Y").strftime("%Y-%m-%d")
        Date=d2


    
    final_list=total_str.split('.')#不改动
    
    
    
 ##################################################################filter order
    if len(beg_target)!=0 and (len(end_target1)!=0 or len(end_target2)!=0) :

        beg_pos=[]
        for i in range(len(beg_target)):
            I=final_list.index(beg_target[i])
            beg_pos.append(I)
        
        end_pos1=[]
        for i in range(len(end_target1)):
            I=final_list.index(end_target1[i])
            end_pos1.append(I)
        end_pos2=[]
        for i in range(len(end_target2)):
            I=final_list.index(end_target2[i])
            end_pos2.append(I)
        
    
######################################################find MDA position in list
        if len(end_pos1)!=0:
            end_pos=end_pos1
            beg_pos0=list(filter(lambda x:x<=end_pos[-1],beg_pos))
            if len(end_target1)>=2:
                end_pos=end_pos[1:]
        else:
            end_pos=end_pos2
            beg_pos0=list(filter(lambda x:x<=end_pos[-1],beg_pos))
            if len(end_target2)>=2:
                end_pos=end_pos[1:]
        if beg_pos0 and end_pos and beg_pos0[-1] < end_pos[0]:
            MDA=final_list[beg_pos0[-1]:end_pos[0]]
            MDA_txt='.'.join(MDA)
            MDA_txt=MDA_txt.translate(remove_digits)###### 去掉数字 
            MDA_txt=MDA_txt.translate(table) ############# 去掉所有标点
            Final_txt.append(MDA_txt)
            word_list=f.split('_')
            year.append(int(word_list[2])-1)  ######int有改动
            Firm.append(word_list[1])
            CIK.append(word_list[0])
            DD.append(Date)
            print('MDA extracted for '+' '+word_list[0]+word_list[1]+' in year '+str(year[-1]))
    else:
        word_list=f.split('_')
        Failed_list.append(f)
        print('no MDA for'+' '+word_list[0]+word_list[1]+' in year '+str(year[-1]))

import pandas as pd
###############################################################save report date
Firm=pd.DataFrame(Firm).T
year=pd.DataFrame(year).T
CIK=pd.DataFrame(CIK).T
DD=pd.DataFrame(DD).T
search_ind=pd.concat([Firm,year,CIK,DD])
search_ind=search_ind.T
search_ind.to_csv('search_index.csv',index=False)
Financial_data=pd.read_csv('fd.csv')

########################################################################### NLP

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
import csv
# load dictionary, convert to list, exchange to lowercase
dic_negative=pd.read_csv('negative.csv')
dic_Positive=pd.read_csv('Positive.csv')
dic_Uncertainty=pd.read_csv('Uncertainty.csv')
dic_Constraining=pd.read_csv('Constraining.csv')
dic_Litigious=pd.read_csv('Litigious.csv')
dic_Modal=pd.read_csv('Modal.csv')

dic_negative_list=dic_negative.iloc[:,0].tolist()
dic_Positive_list=dic_Positive.iloc[:,0].tolist()
dic_Uncertainty_list=dic_Uncertainty.iloc[:,0].tolist()
dic_Constraining_list=dic_Constraining.iloc[:,0].tolist()
dic_Litigious_list=dic_Litigious.iloc[:,0].tolist()
dic_Modal_list=dic_Modal.iloc[:,0].tolist()

Dic_negative_list=[]
Dic_negative_list=[]
Dic_Positive_list=[]
Dic_Uncertainty_list=[]
Dic_Constraining_list=[]
Dic_Litigious_list=[]
Dic_Modal_list=[]

#change case
for d in dic_negative_list:
    if isinstance(d,float):
        print(d)
    else:
        Dic_negative_list.append(d.lower())
        
for d in dic_Positive_list:
    if isinstance(d,float):
        print(d)
    else:
        Dic_Positive_list.append(d.lower())
        
for d in dic_Uncertainty_list:
    if isinstance(d,float):
        print(d)
    else:
        Dic_Uncertainty_list.append(d.lower())
        
for d in dic_Constraining_list:
    if isinstance(d,float):
        print(d)
    else:
        Dic_Constraining_list.append(d.lower())

for d in dic_Litigious_list:
    if isinstance(d,float):
        print(d)
    else:
        Dic_Litigious_list.append(d.lower())

for d in dic_Modal_list:
    if isinstance(d,float):
        print(d)
    else:
        Dic_Modal_list.append(d.lower())

##### match words with dicionary
Filtered_list=[]
I=1
Filtered_list_negative=[]
Filtered_list_positive=[]
Filtered_list_uncertainty=[]
Filtered_list_constraining=[]
Filtered_list_litigious=[]
Filtered_list_modal=[]
n_negative=[]
n_positive=[]
n_uncertainty=[]
n_constraining=[]
n_litigious=[]
n_modal=[]
n_filtered0=[]

for t in Final_txt:
    single_wordlist=word_tokenize(t)
    filtered=[w for w in single_wordlist if not w in stop_words]
    filtered0=[]
    filtered_negative=[]
    filtered_positive=[]
    filtered_uncertainty=[]
    filtered_constraining=[]
    filtered_litigious=[]
    filtered_modal=[]
    #case-insensitive
    for d in filtered:
        if isinstance(d,str):
            filtered0.append(d.lower())
    
    n_filtered0.append(len(filtered0))  
    
    filtered_negative=[x for x in filtered0 if x in Dic_negative_list] #get filtered words from negative dictionary
    Filtered_list_negative.append(' '.join(filtered_negative)) # join all filtered words in "t" txt as a string
    n_negative.append(len(filtered_negative))# get number of filtered negative words in this txt
    
    filtered_positive=[x for x in filtered0 if x in Dic_Positive_list] 
    Filtered_list_positive.append(' '.join(filtered_positive))
    n_positive.append(len(filtered_positive))
    
    filtered_uncertainty=[x for x in filtered0 if x in Dic_Uncertainty_list] 
    Filtered_list_uncertainty.append(' '.join(filtered_uncertainty))
    n_uncertainty.append(len(filtered_uncertainty))
    
    filtered_constraining=[x for x in filtered0 if x in Dic_Constraining_list] 
    Filtered_list_constraining.append(' '.join(filtered_constraining))
    n_constraining.append(len(filtered_constraining))
    
    filtered_litigious=[x for x in filtered0 if x in Dic_Litigious_list] 
    Filtered_list_litigious.append(' '.join(filtered_litigious))
    n_litigious.append(len(filtered_litigious))
    
    filtered_modal=[x for x in filtered0 if x in Dic_Modal_list] 
    Filtered_list_modal.append(' '.join(filtered_modal))
    n_modal.append(len(filtered_modal))
    
    I=I+1
    print(I)

#import numpy as np  #save dictionary to local
#np.save('my_dic_html.npy',Filtered_dic)
#read dictionary method below
#read_dictionary = np.load('my_dic.npy').item()
#save list to local

#import pickle
#with open("test-719.txt", "wb") as fp:   #Pickling
#    pickle.dump(Filtered_list, fp)
#read
#with open("test-719.txt", "rb") as fp:
#    read_list = pickle.load(fp)
#read_list
################################################################ vectorization and TF-IDF 

#arr=np.delete(arr, 0, 0)

from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer
# TFIDF on all variables
vectorizer=CountVectorizer()
transformer = TfidfTransformer()
tfidf_negative = transformer.fit_transform(vectorizer.fit_transform(Filtered_list_negative)) #run tf-idf for list of negative words
feature_name1 = vectorizer.get_feature_names()# get all unique words

vectorizer=CountVectorizer()
transformer = TfidfTransformer()
tfidf_positive = transformer.fit_transform(vectorizer.fit_transform(Filtered_list_positive))
feature_name2= vectorizer.get_feature_names()

vectorizer=CountVectorizer()
transformer = TfidfTransformer()
tfidf_uncertainty = transformer.fit_transform(vectorizer.fit_transform(Filtered_list_uncertainty))
feature_name3= vectorizer.get_feature_names()

vectorizer=CountVectorizer()
transformer = TfidfTransformer()
tfidf_constraining = transformer.fit_transform(vectorizer.fit_transform(Filtered_list_constraining))
feature_name4= vectorizer.get_feature_names()

vectorizer=CountVectorizer()
transformer = TfidfTransformer()
tfidf_litigious = transformer.fit_transform(vectorizer.fit_transform(Filtered_list_litigious))
feature_name5= vectorizer.get_feature_names()

vectorizer=CountVectorizer()
transformer = TfidfTransformer()
tfidf_modal= transformer.fit_transform(vectorizer.fit_transform(Filtered_list_modal))
feature_name6= vectorizer.get_feature_names()
#
import numpy as np ############################## how to adjust
TFIDF_negative=tfidf_negative.toarray()
Xnegative=np.sum(TFIDF_negative,axis=1)*n_negative/n_filtered0  # total negative words frequency (variable one)
TFIDF_positive=tfidf_positive.toarray()
Xpositive=np.sum(TFIDF_positive,axis=1)*n_positive/n_filtered0
TFIDF_uncertainty=tfidf_uncertainty.toarray()
Xuncertainty=np.sum(TFIDF_uncertainty,axis=1)*n_uncertainty/n_filtered0
TFIDF_constraining=tfidf_constraining.toarray()
Xconstraining=np.sum(TFIDF_constraining,axis=1)*n_constraining/n_filtered0
TFIDF_litigious=tfidf_litigious.toarray()
Xlitigious=np.sum(TFIDF_litigious,axis=1)*n_litigious/n_filtered0
TFIDF_modal=tfidf_modal.toarray()
Xmodal=np.sum(TFIDF_modal,axis=1)*n_modal/n_filtered0


############################### code tested above ###########################

################################################################## RNN
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import math, random

Xnegative=Xnegative.reshape(-1,1)
Xpositive=Xpositive.reshape(-1,1)
Xuncertainty=Xuncertainty.reshape(-1,1)
Xconstraining=Xconstraining.reshape(-1,1)
Xlitigious=Xlitigious.reshape(-1,1)
Xmodal=Xmodal.reshape(-1,1)

matrix=np.concatenate((Xnegative,Xpositive,Xuncertainty,Xconstraining,Xlitigious,Xmodal),axis=1)       
matrix=matrix.transpose()    
torch.manual_seed(1)    
M=torch.tensor(matrix)

def sine_2(X, signal_freq=60.):
    return (np.sin(2 * np.pi * (X) / signal_freq) + np.sin(4 * np.pi * (X) / signal_freq)) / 2.0

def noisy(Y, noise_range=(-0.05, 0.05)):
    noise = np.random.uniform(noise_range[0], noise_range[1], size=Y.shape)
    return Y + noise

def sample(sample_size):
    random_offset = random.randint(0, sample_size)
    X = np.arange(sample_size)
    Y = noisy(sine_2(X + random_offset))
    return Y

class SimpleRNN(nn.Module):
    def __init__(self, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(1, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, 2, dropout=0.05)
        self.out = nn.Linear(hidden_size, 1)

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden

n_epochs = 100
n_iters = 50
hidden_size = 10

model = SimpleRNN(hidden_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

losses = np.zeros(n_epochs) # For plotting

for epoch in range(n_epochs):

    for iter in range(n_iters):
        _inputs = sample(50)  ###################修改变量和y值
        inputs = Variable(torch.from_numpy(_inputs[:-1]).float())
        targets = Variable(torch.from_numpy(_inputs[1:]).float())

        # Use teacher forcing 50% of the time
        force = random.random() < 0.5
        outputs, hidden = model(inputs, None, force)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses[epoch] += loss.data[0]

    if epoch > 0:
        print(epoch, loss.data[0])
        
hidden = None#修改？


outputs, hidden = model(inputs, hidden)   #预测

