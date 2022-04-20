#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 13:31:15 2022

@author: ray
"""



import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import time
import requests


#%%%%%
""""For Crypto data"""

"""
Date setting
"""
from_thisdate = "2017-01-01"
start_struct_time =time.strptime(from_thisdate, "%Y-%m-%d")
start_date = int(time.mktime(start_struct_time))

end_thisdate = "2022-01-01"
end_struct_time =time.strptime(end_thisdate, "%Y-%m-%d")
end_date = int(time.mktime(end_struct_time))

"""Crypto type"""

currency1 = 'USDT_BTC'
currency2 = 'USDT_ETH'




"""Crawling"""

res2 = requests.get(('https://poloniex.com/public?command=returnChartData&currencyPair={0}&start={1}&end={2}&period=86400').format(currency1,start_date,end_date,)) 
res3 = requests.get(('https://poloniex.com/public?command=returnChartData&currencyPair={0}&start={1}&end={2}&period=86400').format(currency2,start_date,end_date,)) 

df2 = pd.DataFrame(res2.json())
df3 = pd.DataFrame(res3.json())

"""timestamp transfer """

df2['date']=pd.to_datetime(df2['date'], unit='s')

df3['date']=pd.to_datetime(df3['date'], unit='s')


"""" Date merge"""

def merge(data):
    
    data['date'] = data['date'].astype(str)
    
    
    
    for i in range(len(data)):
        
        data['date'][i] = data['date'][i][:10]
    
    return data


merge(df2)
merge(df3)


df2 = df2.drop(0).reset_index(drop = True)
df3 = df3.drop(0).reset_index(drop = True)



btc = pd.DataFrame()
btc['date'] = df2['date']
btc['tic'] = 'BTC'
btc['close']= df2['close']
btc['high'] = df2['high']
btc['low'] = df2['low']
btc['open'] = df2['open']
btc['volume'] = df2['volume']

eth = pd.DataFrame()
eth['date'] = df3['date']
eth['tic'] = 'ETH'
eth['close']= df3['close']
eth['high'] = df3['high']
eth['low'] = df3['low']
eth['open'] = df3['open']
eth['volume'] = df3['volume']


crypto = pd.concat([btc,eth]).sort_values(['date'])
   
#%%%%%% 



"""
For Stock fata
"""

asset_list = ['SPY', 'IVV', 'VTI', 'VOO', 'QQQ', 'BTC', 'ETH', 'USDC-USD']
ticker_list = ['SPY', 'IVV', 'VTI', 'VOO', 'QQQ', 'USDC-USD']

df = pdr.get_data_yahoo(ticker_list, 
                          start='2017-01-01', end="2022-01-01")


data = df.copy()

data = data.stack().reset_index()
data.columns.names = [None]
data = data.drop(['Close'], axis=1)
data.columns = ['date','tic','close','high','low','open','volume']
data.date = data.date.astype(str)


#%%


"""
Data Combination
"""


asset = pd.concat([data, crypto]).sort_values(['date', 'tic']).reset_index(drop = True)
#print(asset)


"""
Check Data Point
"""
no_datasets = []
for i in asset_list:
    no_data_points = asset[asset['tic']==i].shape[0]
    no_datasets.append((i,no_data_points))
    data_points_df = pd.DataFrame(no_datasets)

#print(data_points_df)
#%%

"""
Reshape data point
"""



date_list1 = list(asset[asset['tic']=='SPY'].date)
date_list2 = list(asset[asset['tic']=='USDC-USD'].date)
asset_filtered = asset[asset['date'].isin(date_list1)]
asset_filtered = asset_filtered[asset_filtered['date'].isin(date_list2)].reset_index(drop =True)
#print(asset_filtered)

"""
Check data point
"""

no_datasets = []
for i in asset_list:
    no_data_points = asset_filtered[asset_filtered['tic']==i].shape[0]
    no_datasets.append((i,no_data_points))
    data_points_df = pd.DataFrame(no_datasets)


print('-'*20, 'Total Data Point', '-'*20)
print(data_points_df)

print('-'*20,'Trading Asset','-'*20)
print(asset_filtered)
print('shape of stocks data input : ' + str(asset_filtered.shape))

asset_filtered.to_csv('/Users/ray/Desktop/This semester/RL/RL_project/asset_data.csv')


