from abc import ABC, abstractmethod

class AbstractPartitioner(ABC):
    
    def __init__(self, array):
        self.array = array
        
    @abstractmethod
    def partition(self):
        pass