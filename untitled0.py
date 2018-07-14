# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 19:05:18 2018

@author: gl452
"""
import pandas as pd
import numpy as np
import urllib3
import warnings
warnings.filterwarnings("ignore")

http = urllib3.PoolManager()

#cik_date = pd.read_csv('comp_cik_date.csv')
cik_date = pd.read_excel('DATA.xls')
# if you read the data.xls, please run code below until cik_date['fyear'] = ....., otherwise you should skip that part
cik_date.columns = cik_date.columns.str.lower()
cik_date['fyear'] = cik_date.date.dt.year
cik_fyear = cik_date[['fyear','cik','tic']]
cik_fyear.drop_duplicates(subset=['fyear','cik'],inplace=True)
cik_fyear.dropna(inplace=True) # Drop those companies without CIK
cik_fyear.cik = cik_fyear.cik.astype(int)
cik_fyear.fyear = cik_fyear.fyear.astype(int)

error_record = pd.DataFrame(columns=['cik','year']) # Check which CIK in which year has no 10-K
no_cik_record = pd.DataFrame(columns=['cik']) # Check which CIK has no matching record

for cik in list(np.unique(cik_fyear.cik)):
    cik_date_temp = cik_fyear[cik_fyear.cik == cik].reset_index()
    tic = cik_date_temp.tic[0]
    web_add_10k = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=' + str(cik) + '&type=10-k&dateb=&owner=exclude&count=100'

    size_table = []
    all_10k = pd.read_html(web_add_10k)
    for n in range(0,np.size(all_10k)):
#        size_table.append(np.size(all_10k[n],0))
        temp_all_10k = all_10k[n]
        if (temp_all_10k.iloc[0,0] == 'Filings') and (temp_all_10k.iloc[0,1] == 'Format'):
            all_10k = all_10k[n]
            all_10k.columns = all_10k.iloc[0,:]
            all_10k.drop([0],axis=0,inplace=True)            
    
    if type(all_10k) is list:
        print('No matching CIK for ' + str(cik))
        no_cik_record = no_cik_record.append(pd.DataFrame([cik],columns=['cik']))
        continue
#        
#    size_table = np.array(size_table)
#    all_10k = all_10k[size_table.argmax()] 
#    all_10k.columns = all_10k.iloc[0,:]
#    all_10k.drop([0],axis=0,inplace=True)
    
    if all_10k.shape[0] != 0:
        all_10k['Filing Date'] = pd.to_datetime(all_10k['Filing Date'],format = '%Y-%m-%d')
        all_10k['acc_num'] = all_10k.Description.str.extract('Acc-no: (.+?)\s')
    
        for year in list(cik_date_temp.fyear):
            ind_year = all_10k['Filing Date'].dt.year == year
            
            try:
                acc_num = all_10k.at[np.where(ind_year)[0][0]+1,'acc_num']
                acc_num_total = ind_year.sum()
                for n in range(0,acc_num_total):
                    acc_num = all_10k.at[np.where(ind_year)[0][n]+1,'acc_num']
                    acc_num_pure_num = acc_num.replace('-','')
                    acc_num_pure_num = str(acc_num_pure_num)
                    web_add_10k_specific = 'https://www.sec.gov/Archives/edgar/data/'+str(cik)+'/'+ acc_num_pure_num +'/'+ acc_num + '-index.htm'
                    table_10k = pd.read_html(web_add_10k_specific)
                    table_10k = table_10k[0]
                    table_10k.columns = table_10k.iloc[0,:]
            #        table_10k.drop([0],inplace=True)
                    
                    ind_10k_file = table_10k.Type.str.contains('10-k',case=False)
                    doc_name = table_10k.at[np.where(ind_10k_file)[0][0],'Document']
                    doc_type = table_10k.at[np.where(ind_10k_file)[0][0],'Type']
                    doc_type = doc_type.replace('/','')
                    if doc_name is np.nan:
                        ind_10k_file = table_10k.Description.str.contains('Complete submission text file',case=False)
                        doc_name = table_10k.at[np.where(ind_10k_file)[0][0],'Document']
                        export_file_name =str(cik)+'_'+ tic + '_' + str(year) + '_'+ doc_type +'_report.txt'
                        writing_method = 'w'
                    elif doc_name.find('.txt') == -1:
                        export_file_name = str(cik)+'_'+tic + '_' + str(year) + '_'+ doc_type +'_report.html'
                        writing_method = 'wb'
                    else:
                        export_file_name = str(cik)+'_'+tic + '_' + str(year) + '_'+ doc_type +'_report.txt'
                        writing_method = 'w'
                    

                    file_add_10k = 'https://www.sec.gov/Archives/edgar/data/' + str(cik) + '/' + acc_num_pure_num + '/' + doc_name
                    
                    r = http.request('get', file_add_10k)
                    if writing_method == 'w':
                        writing_content = r.data.decode('utf-8')
                    else:
                        writing_content = r.data
                    
                    with open(export_file_name, writing_method) as fid:
                        fid.write(writing_content)
                    
                    print('Have saved ' + export_file_name)
            
            except:
                print('No file for '+str(cik)+ ' in ' + str(year))
                err_temp = {'cik': [cik], 'year': [year]}
                error_record = error_record.append(pd.DataFrame(data=err_temp))
                pass
            
"""
Created on Fri Jul  6 10:36:44 2018

@author: RG
"""
#################################################################remove tags
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
table = str.maketrans({key: None for key in string.punctuation})

remove_digits = str.maketrans('', '', digits)

def filter_tags(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
    re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
    re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
    re_br=re.compile('<br\s*?/?>')#处理换行
    re_h=re.compile('</?\w+[^>]*>')#HTML标签
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('\n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    blank_line=re.compile('\n+')
    s=blank_line.sub('\n',s)
    s=replaceCharEntity(s)#替换实体
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
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
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
###############################################################################


import os
path="...." #path
files=os.listdir(path)
Final_txt=[]
year=[]
Firm=[]
Failed_list=[]
final_list=[]
CIK=[]
    
for i in files:
    add='......' +str(i)   #path of file
    page=urlopen(add)
    contents = page.read() 
    soup=BeautifulSoup(contents,'lxml')
    text=soup.get_text()
    
    clean_text=Cleaning_data(text)
    text_list = clean_text.split('.')
    clean_list=[txt.replace('\n','') for txt in text_list]
    clean_list=[txt.replace('\xa0','') for txt in clean_list]
    clean_list0=[txt.strip() for txt in clean_list]  #remove back and front vacancy
    
    total_str='.'.join(clean_list0)
    total_str = '.' + total_str
    
    a=re.compile(r'([^.]*?discussion[^.]*?and[^.]*?analysis[^.]*?of[^.]*?financial[^.]*?condition[^.]*?results[^.]*?operations[^.]*\.)', re.IGNORECASE)
    beg_target=a.findall(total_str)
    beg_target=[ori.replace('.','') for ori in beg_target]
    
    b=re.compile(r'([^.]*?and[^.]*?qualitative[^.]*?Disclosures[^.]*?about[^.]*?market[^.]*\.)', re.IGNORECASE)
    end_target1=b.findall(total_str)
    end_target1=[ori.replace('.','') for ori in end_target1]
    
    d=re.compile(r'([^.]*?FINANCIAL[^.]*?STATEMENTS[^.]*?and[^.]*?SUPPLEMENTARY[^.]*?data[^.]*\.)', re.IGNORECASE)
    end_target2=d.findall(total_str)
    end_target2=[ori.replace('.','') for ori in end_target2]  
        
    final_list=total_str.split('.')
 ###############################################################################################################
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
  
        #split again, to match target element
####################################################### find MDA position in list
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
  
        MDA=final_list[beg_pos0[-1]:end_pos[0]]
        MDA_txt='.'.join(MDA)
        MDA_txt=MDA_txt.translate(remove_digits)###### 去掉数字 
        MDA_txt=MDA_txt.translate(table) ############# 去掉所有标点
        Final_txt.append(MDA_txt)
        word_list=add.split('_')
        year.append(int(word_list[2])-1) 
        Firm.append(word_list[1])
        CIK.append(word_list[0])
        print('MDA extracted for '+' '+word_list[0]+word_list[1]+' in year '+(int(word_list[2])-1))
    else:
        word_list=add.split('_')
        Failed_list.append(i)
        print('no MDA for'+' '+word_list[0]+word_list[1]+' in year '+(int(word_list[2])-1))
