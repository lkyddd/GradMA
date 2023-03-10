from abc import ABC, abstractmethod


class BaseRunner(ABC):
    @abstractmethod
    def run(self):
        pass
