from abc import ABC, abstractmethod


class BaseContext(ABC):

    def __init__(self, role_name_size_list, rank):
        self.role_size_dict = dict(role_name_size_list)
        self.worker_size = sum([size for _, size in role_name_size_list])
        self.rank = rank

        self.role_name = ""
        self.role_index = 0
        self.__init_role__(role_name_size_list)

    def __init_role__(self, role_name_size_list):
        expended_role_index = [(role_name, role_index) for role_name, role_size in role_name_size_list
                               for role_index in range(role_size)]
        self.role_name, self.role_index = expended_role_index[self.rank]

    def get_role_size(self, role_name):
        return self.role_size_dict[role_name]

    @abstractmethod
    def get_node(self, role_name, role_index=0):
        pass

    @abstractmethod
    def get_node_list(self, role_name):
        pass

    @abstractmethod
    def shutdown_cluster(self):
        pass

    @abstractmethod
    def barrier(self):
        pass