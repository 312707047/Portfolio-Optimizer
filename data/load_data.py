import itertools
import logging
import crypto
import os

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from multiprocessing import Pool


class BaseDataLoader(ABC):
    def __init__(self,
                 tag:list,
                 kline_size:str='15m'):
        
        self.tag = tag
        self.kline_size = kline_size
    
    @abstractmethod
    def load_data(self, *arg):
        pass


class RawDataLoader(BaseDataLoader):
    '''Download kline data from binance and preprocess them into a three dimensional array (High, Low, Close)'''
    def __init__(self,
                 tag:list,
                 kline_size:str='15m'):
        
        super().__init__(tag, kline_size)
        self.__get_data()
        
    def __get_data(self):
        logger = logging.getLogger()
        
        # preprocess tag to get the data
        self.new_tag = [tag+'USDT' for tag in self.tag]
        tag_with_klines = list(itertools.zip_longest(self.new_tag, [self.kline_size], fillvalue=self.kline_size))
        
        # get the data with multiprocessing and Binance API
        logger.info('Downloading data...')
        
        with Pool() as pool:
            pool.starmap(crypto.get_all_binance, tag_with_klines)
        
        logger.info('Finished download!')
    
    def load_data(self, data_path):
        logger = logging.getLogger()
        logger.info('Loading data...')
        
        files = os.listdir(data_path)
        data = pd.concat(
            (pd.read_csv(f'{data_path}/{file}', parse_dates=True, index_col=['Timestamp']) for file in files),
            join='inner',
            axis=1)
        
        data.drop(['Close_time', 'Ignore', 'Quote_av', 'Tb_base_av', 'Tb_quote_av'], axis=1, inplace=True)
        
        logger.info('Finished loading data!')
        return data