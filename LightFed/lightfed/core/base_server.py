from abc import ABC, abstractmethod


class BaseServer(ABC):

    def __init__(self, context):
        self._ct_ = context
        self._role_name_ = context.role_name
        self._role_index_ = context.role_index

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def end(self):
        pass

    @abstractmethod
    def end_condition(self):
        pass
