# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:36:44 2018

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
path="C:/Users/'.....'/Desktop/capstone/html_1"
files=os.listdir(path)


Final_txt=[]
year=[]
Firm=[]
Failed_list=[]
final_list=[]
CIK=[]
DD=[]  


for f in files:
    add= 'file:///C:/Users/'.....'/Desktop/capstone/html_1/'+f
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
    dd_target_1=dd_1.findall(total_str)
    if dd_target_1:
        dd_target_1=dd_target_1[0]
        dd_target_1=dd_target_1.translate(table) 
        dd_target_1=dd_target_1.upper()
        dd_target_1=dd_target_1.replace(" ","")
        dd_target_1=dd_target_1.replace("\t","")
        date1=re.findall(r"ENDED[A-Z]{3,8}\d{6}|NUMBER[A-Z]{3,8}\d{6}",dd_target_1)
        
        if date1:
            date1 = date1[0]
            if date1.startswith('E'):
                 date1=date1[5:]
            else:
                 date1=date1[6:]
            Date=datetime.strptime(date1,"%B%d%Y").strftime("%Y-%m-%d")
        else:
            continue
        

    else:
        print("This is not a valid text", f)
        continue
    
    final_list=total_str.split('.')
    
    
    
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
###############################################################################
        
#write
filename='MDA_TXT.txt'

with open(filename,'w', encoding = 'utf-8') as file:
    for L in Final_txt:
        file.write(L+'\n')
file.close()

#read
filename='MDA_HTML_1.txt'
Final_txt=[]
with open(filename,'r') as file:
    for cand in file.readlines():
        #string=cand.strip()
        Final_txt.append(cand.strip())
    

# save as dataframe
import pandas as pd

Firm=pd.DataFrame(Firm).T
year=pd.DataFrame(year).T
CIK=pd.DataFrame(CIK).T
DD=pd.DataFrame(DD).T
search_ind=pd.concat([Firm,year,CIK,DD])
search_ind=search_ind.T
search_ind.to_csv('SearchIndex_TXT.csv',index=False)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 ##################################################################################################################################################################       
path2="C:/Users/'....'/Desktop/capstone/TXT"
files_2=os.listdir(path2)
Final_txt=[]
year=[]
Firm=[]
Failed_list=[]
final_list=[]
CIK=[]
DD=[] 



for u in files_2:
    add= 'file:///C:/Users/'.....'/Desktop/capstone/TXT/'+u
    text=open(u, 'r').read()
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
        
    
    dd_1=re.compile(r'([^.]*?for[^.]*?the[^.]*?year[^.]*?ended[^.]*\.)', re.IGNORECASE)########################################################### 修改
    dd_target_1=dd_1.findall(total_str)
    if dd_target_1:
        dd_target_1 = dd_target_1[0]
        dd_target_1=dd_target_1.translate(table) 
        dd_target_1=dd_target_1.upper()
        dd_target_1=dd_target_1.replace(" ","")
        dd_target_1=dd_target_1.replace("\t","")
        date1=re.findall(r"ENDED[A-Z]{3,8}\d{6}|NUMBER[A-Z]{3,8}\d{6}",dd_target_1)
        if date1:
            date1 = date1[0]
            if date1.startswith('E'):
                date1=date1[5:]
            else:
                date1=date1[6:]
            Date=datetime.strptime(date1,"%B%d%Y").strftime("%Y-%m-%d")
        else:
            continue
    else:
        print("This is not a valid txt",u)
        continue
                
    
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
            word_list=u.split('_')
            year.append(int(word_list[2])-1)  ######int有改动
            Firm.append(word_list[1])
            CIK.append(word_list[0])
            DD.append(Date)
            print('MDA extracted for '+' '+word_list[0]+word_list[1]+' in year '+str(year[-1]))
    else:
        word_list=u.split('_')
        Failed_list.append(u)
        print('no MDA for'+' '+word_list[0]+word_list[1]+' in year '+str(year[-1]))
###############################################################################
import pandas as pd

Firm=pd.DataFrame(Firm).T
year=pd.DataFrame(year).T
CIK=pd.DataFrame(CIK).T
DD=pd.DataFrame(DD).T
search_ind=pd.concat([Firm,year,CIK,DD])
search_ind=search_ind.T
search_ind.to_csv('search_index.csv',index=False)
Financial_data=pd.read_csv('fd.csv')

filename='MDA_HTML_1.txt'
text1=[]
with open(filename,'r',encoding = 'utf-8') as file:
    for cand in file.readlines():
        text1.append(cand.strip())

filename='MDA_HTML_2.txt'
text2=[]
with open(filename,'r',encoding = 'utf-8') as file:
    for cand in file.readlines():
        text2.append(cand.strip())

filename='MDA_HTML_3.txt'
text3=[]
with open(filename,'r',encoding = 'utf-8') as file:
    for cand in file.readlines():
        text3.append(cand.strip())
        
filename='MDA_HTML_4.txt'
text4=[]
with open(filename,'r',encoding = 'utf-8') as file:
    for cand in file.readlines():
        text4.append(cand.strip())

filename='MDA_TXT.txt'
text=[]
with open(filename,'r',encoding = 'utf-8') as file:
    for cand in file.readlines():
        text.append(cand.strip())

Final_txt=text1+text2+text3+text4+text

filename='FINAL_TXT.txt'

with open(filename,'w', encoding = 'utf-8') as file:
    for L in Final_txt:
        file.write(L+'\n')
file.close()

ind1=pd.read_csv('SearchIndex_html_1.csv')
ind2=pd.read_csv('SearchIndex_html_2.csv')
ind3=pd.read_csv('SearchIndex_html_3.csv')
ind4=pd.read_csv('SearchIndex_html_4.csv')
indt=pd.read_csv('SearchIndex_TXT.csv')

IND=ind1.append(ind2)
IND=IND.append(ind3)
IND=IND.append(ind4)
IND=IND.append(indt)

IND.to_csv('report_date.csv',index=False)
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
I=0
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
    filtered=[]
    filtered_negative=[]
    filtered_positive=[]
    filtered_uncertainty=[]
    filtered_constraining=[]
    filtered_litigious=[]
    filtered_modal=[]
    t=t.lower()
    single_wordlist=word_tokenize(t)
    filtered=[w for w in single_wordlist if not w in stop_words]


    #case-insensitive
    #for d in filtered:
        #if isinstance(d,str):
            #filtered0.append(d.lower())
    
    n_filtered0.append(len(filtered))  
    
    filtered_negative=[x for x in filtered if x in Dic_negative_list] #get filtered words from negative dictionary
    n_negative.append(len(filtered_negative))# get number of filtered negative words in this txt
    filtered_negative=' '.join(filtered_negative)
    Filtered_list_negative.append(filtered_negative) # join all filtered words in "t" txt as a string

    
    filtered_positive=[x for x in filtered if x in Dic_Positive_list]
    n_positive.append(len(filtered_positive))
    filtered_positive=' '.join(filtered_positive)
    Filtered_list_positive.append(filtered_positive)
    
    
    filtered_uncertainty=[x for x in filtered if x in Dic_Uncertainty_list] 
    n_uncertainty.append(len(filtered_uncertainty))
    filtered_uncertainty=' '.join(filtered_uncertainty)
    Filtered_list_uncertainty.append(filtered_uncertainty)
    
    filtered_constraining=[x for x in filtered if x in Dic_Constraining_list]
    n_constraining.append(len(filtered_constraining))
    filtered_constraining=' '.join(filtered_constraining)
    Filtered_list_constraining.append(filtered_constraining)
    
    
    filtered_litigious=[x for x in filtered if x in Dic_Litigious_list] 
    n_litigious.append(len(filtered_litigious))
    filtered_litigious=' '.join(filtered_litigious)
    Filtered_list_litigious.append(filtered_litigious)
    
    
    filtered_modal=[x for x in filtered if x in Dic_Modal_list] 
    n_modal.append(len(filtered_modal))
    filtered_modal=' '.join(filtered_modal)
    Filtered_list_modal.append(filtered_modal)
    
    
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
n_filtered0=np.asarray(n_filtered0)

TFIDF_negative=tfidf_negative.toarray()
#Xnegative=np.sum(TFIDF_negative,axis=1)*n_negative/n_filtered0  # total negative words frequency (variable one)
Xnegative=np.sum(TFIDF_negative,axis=1)
n_negative=np.asarray(n_negative)
Xnegative_pro=n_negative/n_filtered0

TFIDF_positive=tfidf_positive.toarray()
#Xpositive=np.sum(TFIDF_positive,axis=1)*n_positive/n_filtered0
Xpositive=np.sum(TFIDF_positive,axis=1)
n_positive=np.asarray(n_positive)
Xpositive_pro=n_positive/n_filtered0

TFIDF_uncertainty=tfidf_uncertainty.toarray()
#Xuncertainty=np.sum(TFIDF_uncertainty,axis=1)*n_uncertainty/n_filtered0
Xuncertainty=np.sum(TFIDF_uncertainty,axis=1)
n_uncertainty=np.asarray(n_uncertainty)
Xuncertainty_pro=n_uncertainty/n_filtered0

TFIDF_constraining=tfidf_constraining.toarray()
#Xconstraining=np.sum(TFIDF_constraining,axis=1)*n_constraining/n_filtered0
Xconstraining=np.sum(TFIDF_constraining,axis=1)
n_constraining=np.asarray(n_constraining)
Xconstraining_pro=n_constraining/n_filtered0

TFIDF_litigious=tfidf_litigious.toarray()
#Xlitigious=np.sum(TFIDF_litigious,axis=1)*n_litigious/n_filtered0
Xlitigious=np.sum(TFIDF_litigious,axis=1)
n_litigious=np.asarray(n_litigious)
Xlitigious_pro=n_litigious/n_filtered0

TFIDF_modal=tfidf_modal.toarray()
#Xmodal=np.sum(TFIDF_modal,axis=1)*n_modal/n_filtered0
Xmodal=np.sum(TFIDF_modal,axis=1)
n_modal=np.asarray(n_modal)
Xmodal_pro=n_modal/n_filtered0

X=pd.DataFrame({'Xnegative':Xnegative,'Xpositive':Xpositive,'Xuncertainty':Xuncertainty,'Xconstraining':Xconstraining,'Xlitigious':Xlitigious,'Xmodal':Xmodal,'Xnegative_pro':Xnegative_pro,'Xpositive_pro':Xpositive_pro,'Xuncertainty_pro':Xuncertainty_pro,'Xconstraining_pro':Xconstraining_pro,'Xlitigious_pro':Xlitigious_pro,'Xmodal_pro':Xmodal_pro})

X.to_csv('xvalue.csv',index=False)

############################### code tested above ###########################

################################################################## RNN

import pandas as pd
X=pd.read_csv('xvalue.csv')
date=pd.read_csv('report_date.csv')
X=pd.concat([X,date],axis=1)
X['fyear']=pd.DatetimeIndex(X['DATE']).year
X['fmonth']=pd.DatetimeIndex(X['DATE']).month
X['fday']=pd.DatetimeIndex(X['DATE']).day
wrds=pd.read_csv('result0817.csv')
date = pd.to_datetime(wrds['DATE'], format = '%Y%m%d')##################
wrds['fyear']=pd.DatetimeIndex(date).year
wrds['fmonth']=pd.DatetimeIndex(date).month
wrds['fday']=pd.DatetimeIndex(date).day
wrds=wrds.dropna(how='any')
wrds.drop_duplicates(keep='first',inplace=True)
wrds=wrds[['TICKER','CIK','BM','SIZE','G','fyear','fmonth','CUM']]

x=pd.merge(X,wrds,how='left',on=['CIK','fyear','fmonth'])
xnew=x.dropna(how='any')
xnew = xnew[~(xnew.isin(['.']))]
xnew=xnew.dropna()

x.to_csv('totalvalue.csv',index=False)
x_output=xnew.drop(columns=['TICKER_y'])
x_output.to_csv('output.csv',index=False)
#################################################################### svm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

x0=xnew[['Xlitigious','Xpositive']]
xs0 = preprocessing.scale(x0,axis=0)
x1=xnew[['Xconstraining_pro','Xlitigious_pro','Xmodal_pro','Xnegative_pro','Xpositive_pro','Xuncertainty_pro']]
xs1 = preprocessing.scale(x1,axis=0)

x2=xnew[['Xconstraining']]
xs2 = preprocessing.scale(x2,axis=0)
x3=xnew[['Xlitigious']]
xs3 = preprocessing.scale(x3,axis=0)
x4=xnew[['Xmodal']]
xs4 = preprocessing.scale(x4,axis=0)
x5=xnew[['Xnegative']]
xs5 = preprocessing.scale(x5,axis=0)
x6=xnew[['Xpositive']]
xs6 = preprocessing.scale(x6,axis=0)
x7=xnew[['Xuncertainty']]
xs7 = preprocessing.scale(x7,axis=0)

x8=xnew[['Xconstraining_pro']]
xs8 = preprocessing.scale(x8,axis=0)
x9=xnew[['Xlitigious_pro']]
xs9 = preprocessing.scale(x9,axis=0)
x10=xnew[['Xmodal_pro']]
xs10 = preprocessing.scale(x10,axis=0)
x11=xnew[['Xnegative_pro']]
xs11 = preprocessing.scale(x11,axis=0)
x12=xnew[['Xpositive_pro']]
xs12 = preprocessing.scale(x12,axis=0)
x13=xnew[['Xuncertainty_pro']]
xs13 = preprocessing.scale(x13,axis=0)

x15=xnew[['Xconstraining','Xlitigious','Xmodal','Xnegative','Xpositive','Xuncertainty','BM','CUM']]
xs15 = preprocessing.scale(x15,axis=0)


y0=xnew['G']
y0=
#pca = PCA(n_components=3)
#x_1=pca.transform(x0)
#scaler = StandardScaler()

#x_std=scaler.fit_transform(x0)
#y_std=scaler.fit_transform(y0[:,np.newaxis])
from sklearn import preprocessing
xs0 = preprocessing.scale(x0)


X_train, X_test, y_train, y_test = train_test_split(xs12,y0,test_size=0.2)

n_samples, n_features = x_scaled.shape[0], x_scaled.shape[1]
clf = SVR(C=1.0,epsilon=0.25,max_iter=500)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

X_train, X_test, y_train, y_test = train_test_split(x_1,y0,test_size=0.25)
n_samples, n_features = x_1.shape[0], x_1.shape[1]
clf = SVR(C=10.0,epsilon=0.2,max_iter=60)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=200 )
regr.fit(X_train, y_train)
regr.score(X_test,y_test)
imp=regr.feature_importances_

from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB()
y_train_nb = y_train.as_matrix()
nbclf.fit(X_train,y_train_nb)
from sklearn.metrics import accuracy_score
y_pred=nbclf.predict(X_test)
accuracy_score(y_test,y_pred)

from sklearn.linear_model import Lasso
laclf = Lasso(fit_intercept = True, normalize = True)
laclf.fit(X_train, y_train)
laclf.score(X_test, y_test)

from sklearn.naive_bayes import MultinomialNB
clf = linear_model.BayesianRidge(n_iter=10)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

X_train, X_test, y_train, y_test = train_test_split(xs12,y0,test_size=0.2)
regr = linear_model.LinearRegression()
r0=regr.fit(X_train,y_train)
r0_coef=pd.DataFrame(r0.coef_)
y0_pred=r0.predict(X_test)
r0_r2_score=r2_score(y_test, y0_pred)
r0_mse=mean_squared_error(y_test, y0_pred)
y0_final_pred=r0.predict(xs12)
r0.score(X_test,y_test)
#est0=sm.OLS(np.asarray(y0),np.asarray(xs0)).fit()
est0=sm.OLS(y0,xs0).fit()
X_train, X_test, y_train, y_test = train_test_split(xs12,y0,test_size=0.2)
clf = SVR(C=10.0,epsilon=0.25,max_iter=300)
clf.fit(X_train, y_train)
clf.score(X_test,y_test)


X_train, X_test, y_train, y_test = train_test_split(xs1,y0,test_size=0.2)
regr = linear_model.LinearRegression()
r1=regr.fit(X_train,y_train)
r1_coef=pd.DataFrame(r1.coef_)
y1_pred=r0.predict(X_test)
r1_r2_score=r2_score(y_test, y1_pred)
r1_mse=mean_squared_error(y_test, y1_pred)
y1_final_pred=r0.predict(xs1)
est1=sm.OLS(y0,xs1).fit()
print(est1.summary())

X_train, X_test, y_train, y_test = train_test_split(xs1,y0,test_size=0.2)
regr = linear_model.LinearRegression()
r1=regr.fit(X_train,y_train)
r1_coef=pd.DataFrame(r1.coef_)
y1_pred=r0.predict(X_test)
r1_r2_score=r2_score(y_test, y1_pred)
r1_mse=mean_squared_error(y_test, y1_pred)
y1_final_pred=r1.predict(xs1)
est2=sm.OLS(y0,xs2).fit()
print(est2.summary())

X_train, X_test, y_train, y_test = train_test_split(xs2,y0,test_size=0.2)
regr = linear_model.LinearRegression()
r2=regr.fit(X_train,y_train)
r2_coef=pd.DataFrame(r2.coef_)
y2_pred=r2.predict(X_test)
r2_r2_score=r2_score(y_test, y2_pred)
r2_mse=mean_squared_error(y_test, y2_pred)
y2_final_pred=r2.predict(xs2)

X_train, X_test, y_train, y_test = train_test_split(xs3,y0,test_size=0.2)
regr = linear_model.LinearRegression()
r3=regr.fit(X_train,y_train)
r3_coef=pd.DataFrame(r3.coef_)
y3_pred=r3.predict(X_test)
r3_r2_score=r2_score(y_test, y3_pred)
r3_mse=mean_squared_error(y_test, y3_pred)
y3_final_pred=r3.predict(xs3)
est3=sm.OLS(y0,xs3).fit()
print(est3.summary())

Xconstraining_pro=pd.DataFrame(Xconstraining_pro)
Xlitigious_pro=pd.DataFrame(Xlitigious_pro)
Xmodal_pro=pd.DataFrame(Xmodal_pro)
Xnegative_pro=pd.DataFrame(Xnegative_pro)
Xpositive_pro=pd.DataFrame(Xpositive_pro)
Xuncertainty_pro=pd.DataFrame(Xuncertainty_pro)
BM=pd.DataFrame(BM)
CUM=pd.DataFrame(CUM)
SIZE=pd.DataFrame(SIZE)
import matplotlib.pyplot as plt

my_x_ticks = np.arange(0, 5, 0.05)
plt.figure(figsize=(20,10))
plt.xticks(my_x_ticks)
plt.xticks(rotation=70)
plt.xlabel('Constraining words frequency')
plt.hist(Xconstraining)
plt.show()
#Xconstraining.plot.hist()
#plt.plot(Xconstraining)

ax = Xuncertainty_pro.plot.hist(xticks= np.arange(0, 0.1, 0.005),grid=True,figsize=(20,10), title='Uncertainty words',fontsize=10,xlim=(0,0.1))
ax.set_xlabel("proportional weight result")
ax.set_ylabel("number")
ax.title.set_size(30)
fig = ax.get_figure()
fig.savefig('pic')


plt.title("Market Cap")
plt.figure(figsize=(20,10))
plt.grid(color='k')
my_x_ticks = np.arange(0, 15000, 500)
plt.xticks(my_x_ticks)
plt.hist(size,bins=30,log=True)
plt.title("Market Cap",fontsize=30)

fig = plt.gcf()
plt.title("Market Cap")
plt.figure(figsize=(20,10))
my_x_ticks = np.arange(0, 15000, 500)
plt.xticks(my_x_ticks)
plt.hist(size,bins=30,log=True)
plt.show()
fig.savefig('he4.png', dpi=100)