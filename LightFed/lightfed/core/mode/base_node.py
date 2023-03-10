from abc import ABC, abstractmethod


class BaseNode(ABC):
    def __init__(self, rank):
        self._rank_ = rank
        self._deepcopy = True

    def set(self, deepcopy=True):
        self._deepcopy = deepcopy
        return self
    
    def _reset(self):
        self._deepcopy = True

    @abstractmethod
    def __getattr__(self, func_name):
        pass
