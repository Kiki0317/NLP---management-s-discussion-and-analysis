# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:15:32 2018

@author: 广芮
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
                    web_add_10k_specific = 'https://www.sec.gov/Archives/edgar/data/'
                    +str(cik)+'/'+ acc_num_pure_num +'/'+ acc_num + '-index.htm'
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
            

            
        
        