####
# This module contains an abstract class to compute beta
####
from abc import ABC, abstractmethod

class Beta(ABC):
    
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def get_beta(self):
        pass        
        
