from mpi4py import MPI
from .cluster_node import ClusterNode
from lightfed.core.mode import BaseContext


class ClusterContext(BaseContext):
    def __init__(self, role_name_size_list, rank):
        super().__init__(role_name_size_list, rank)
        self.comm = MPI.COMM_WORLD

        self.nodes = {}  # {role_name1:node_list1,role_name2:node_list2,...}
        self.__init_nodes__(role_name_size_list)

    def get_node(self, role_name, role_index=0):
        return self.nodes[role_name][role_index]

    def get_node_list(self, role_name):
        return self.nodes[role_name]

    def shutdown_cluster(self):
        for rank in range(self.worker_size):
            self.comm.isend(("__shutdown__", None, None), dest=rank)

    def barrier(self):
        self.comm.barrier()

    def __init_nodes__(self, role_name_size_list):
        rank = 0
        for role_name, role_size in role_name_size_list:
            for _ in range(role_size):
                _node_ = ClusterNode(rank)
                rank += 1
                self.nodes.setdefault(role_name, []).append(_node_)
