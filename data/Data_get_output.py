#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 13:31:15 2022

@author: ray
"""



import pandas as pd
from pandas_datareader import data as pdr
import time
import requests




class GetCryptoData():
    
    def __init__(self, start,end, currency):
        
        from_thisdate = start
        start_struct_time =time.strptime(from_thisdate, "%Y-%m-%d")
        start_date = int(time.mktime(start_struct_time))

        end_thisdate = end
        end_struct_time =time.strptime(end_thisdate, "%Y-%m-%d")
        end_date = int(time.mktime(end_struct_time))
        
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        
        self.res = requests.get(('https://poloniex.com/public?command=returnChartData&currencyPair={0}&start={1}&end={2}&period=86400').format(self.currency,self.start_date,self.end_date,))
    
    def MakeOutPut(self):
        
        df = pd.DataFrame(self.res.json())
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df['date'] = df['date'].astype(str)
    
        for i in range(len(df)):
            
            df['date'][i] = df['date'][i][:10]
        
        
        df = df.drop(0).reset_index(drop = True)
        
        output = pd.DataFrame({'date':df.date, 'tic': self.currency[5:8], 'close':df.close, 'high':df.high,
                               'low': df.low, 'open':df.open, 'volume':df.volume})


        return output





class GetStockData():
    
    def __init__(self, start, end, ticker_list):
        
        self.start = start
        self.end = end
        self.ticker_list = ticker_list
                
    def MakeOutPut(self):
        data = pdr.get_data_yahoo(self.ticker_list, self.start, self.end)
        data = data.stack().reset_index()
        data.columns.names = [None]
        data = data.drop(['Close'], axis=1)
        data.columns = ['date','tic','close','high','low','open','volume']
        data.date = data.date.astype(str)
        
        
        return data    





class GetRawData():
    
    def __init__(self, asset_list, asset_data):
        
        self.asset_list = asset_list
        self.asset_data = asset_data

    
    def MakeOutPut(self):
        
        date_list1 = list(self.asset_data[self.asset_data['tic']=='SPY'].date)
        date_list2 = list(self.asset_data[self.asset_data['tic']=='USDT-USD'].date)
        asset_filtered = self.asset_data[self.asset_data['date'].isin(date_list1)]
        asset_filtered = asset_filtered[asset_filtered['date'].isin(date_list2)].reset_index(drop =True)
        
        print('-'*20,'Trading Asset','-'*20)
        print(asset_filtered)
        
        
        no_datasets = []
        for i in self.asset_list:
            no_data_points = asset_filtered[asset_filtered['tic']==i].shape[0]
            no_datasets.append((i,no_data_points))
            data_points_df = pd.DataFrame(no_datasets)
            
        print('-'*20, 'Total Data Point', '-'*20)
        print(data_points_df)
        
        return asset_filtered
    
    
    
    
class GetFeatures():
    
    def __init__(self, raw_data, feature):
        
        self.raw_data = raw_data
        
        self.feature = feature
        
        
    
    def MakeOutPut(self):
        
        df_prices = self.raw_data.reset_index().set_index(['tic', 'date']).sort_index()

        # Get the list of all the tickers
        tic_list = list(set([i for i,j in df_prices.index]))


        df_prices = df_prices.reset_index().set_index(['tic', 'date']).sort_index()

        # Get all the Close Prices
        df_feature = pd.DataFrame()

        for ticker in tic_list:
            series = df_prices.xs(ticker)[self.feature]
            df_feature[ticker] = series
            
        df_feature = df_feature.reset_index()
        df_feature = df_feature.reindex(columns = ['date','BTC', 'ETH','USDT-USD','SPY', 'IVV', 'VTI', 'VOO', 'QQQ'])

    
        
        return df_feature
    
    def StoreData(self, path):
        
        store_data = self.MakeOutPut()
        store_data.to_csv(path, index = False)
    
    




    
#%%
"""
Data Generating
"""


""" Crypto Data"""

btc = GetCryptoData('2015-01-01', '2022-01-01', 'USDT_BTC').MakeOutPut()
eth = GetCryptoData('2015-01-01', '2022-01-01', 'USDT_ETH').MakeOutPut()
crypto_data = pd.concat([btc,eth]).sort_values(['date'])


""" Stocks Data """
need_stock_list = ['SPY', 'IVV', 'VTI', 'VOO', 'QQQ', 'USDT-USD']
stocks_data = GetStockData('2015-01-01', '2021-12-31', need_stock_list).MakeOutPut()



asset = pd.concat([stocks_data, crypto_data]).sort_values(['date', 'tic']).reset_index(drop = True)



""" Total Raw Data"""

total_assets = ['SPY', 'IVV', 'VTI', 'VOO', 'QQQ', 'BTC', 'ETH', 'USDT-USD']
Raw = GetRawData(total_assets, asset).MakeOutPut()

""" Feature data"""

High = GetFeatures(Raw, 'high').MakeOutPut()
Close = GetFeatures(Raw, 'close').MakeOutPut()
Low = GetFeatures(Raw, 'low').MakeOutPut()


#



























