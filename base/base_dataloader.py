from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    def __init__(self,
                 tag:list,
                 kline_size:str='15m'):
        
        self.tag = tag
        self.kline_size = kline_size
    
    @abstractmethod
    def load_data(self, *arg):
        pass