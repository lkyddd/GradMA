import logging
# from mpi4py import MPI
import setproctitle
# from lightfed.core.mode.cluster import ClusterContext, ClusterRunner
from lightfed.core.mode.local import LocalContext, LocalRunner

CLUSTER_MODE = 1
LOCAL_MODE = 0


class Config:
    def __init__(self):
        self._name_size_func_list_ = []
        self._added_role_size_ = 0
        # self._worker_size_ = MPI.COMM_WORLD.Get_size()
        # self._mode_ = CLUSTER_MODE if self._worker_size_ > 1 else LOCAL_MODE
        self._mode_ = 0

        self._cached_current_context_of_cluster_mode_ = None
        self._cached_context_list_of_local_mode_ = None


    def add_role(self, role_name, role_size, new_server_manager_func):
        # if (self._mode_ == CLUSTER_MODE) and (self._added_role_size_ + role_size > self._worker_size_):
        #     raise Exception("summing of cluster role size is greater than cluster size")
        self._name_size_func_list_.append((role_name, role_size, new_server_manager_func))
        self._added_role_size_ += role_size
        return self

    def get_runner(self):
        role_name_func_dict = {name: func for name, _, func in self._name_size_func_list_}

        # if self._mode_ == CLUSTER_MODE:
        #     context = self._get_current_context_of_cluster_mode_()
        #     func = role_name_func_dict[context.role_name]
        #     server = func(context)
        #     return ClusterRunner(server)

        if self._mode_ == LOCAL_MODE:
            server_list = []
            i = 0
            for context in self._get_context_list_of_local_mode_():
                i = i + 1
                print('i', i)
                func = role_name_func_dict[context.role_name]
                server = func(context)
                server_list.append(server)
            return LocalRunner(server_list)

        return None

    def init_log(self, level=logging.INFO):
        # if self._mode_ == CLUSTER_MODE:
        #     ct = self._get_current_context_of_cluster_mode_()
        #     logging.basicConfig(level=level,
        #                         format=f'r{ct.rank}:{ct.role_name}_{ct.role_index} '
        #                                f'%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        #                         datefmt='%Y%m%d_%H:%M:%S')

        if self._mode_ == LOCAL_MODE:
            logging.basicConfig(level=level,
                                format=f'%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                                datefmt='%Y%m%d_%H:%M:%S')
        return self

    def set_proc_title(self, job_name):
        str_process_name = None
        # if self._mode_ == CLUSTER_MODE:
        #     ec = self._get_current_context_of_cluster_mode_()
        #     str_process_name = f"{job_name}:{ec.role_name}-{ec.role_index}"
        if self._mode_ == LOCAL_MODE:
            str_process_name = job_name
        setproctitle.setproctitle(str_process_name)
        return self

    # def _get_current_context_of_cluster_mode_(self):
    #     if self._mode_ != CLUSTER_MODE:
    #         return None
    #     if self._cached_current_context_of_cluster_mode_ is not None:
    #         return self._cached_current_context_of_cluster_mode_
    #     if self._added_role_size_ != self._worker_size_:
    #         raise Exception("summing of cluster role size is not equal to worker size")
    #
    #     role_name_size_list = [(name, size) for name, size, _ in self._name_size_func_list_]
    #     rank = MPI.COMM_WORLD.Get_rank()
    #     self._cached_current_context_of_cluster_mode_ = ClusterContext(role_name_size_list, rank)
    #     return self._cached_current_context_of_cluster_mode_

    def _get_context_list_of_local_mode_(self):
        if self._mode_ != LOCAL_MODE:
            return None
        if self._cached_context_list_of_local_mode_ is not None:
            return self._cached_context_list_of_local_mode_
        role_name_size_list = [(name, size) for name, size, _ in self._name_size_func_list_]
        self._cached_context_list_of_local_mode_ = []
        for rank in range(self._added_role_size_):
            ct = LocalContext(role_name_size_list, rank)
            self._cached_context_list_of_local_mode_.append(ct)
        return self._cached_context_list_of_local_mode_
