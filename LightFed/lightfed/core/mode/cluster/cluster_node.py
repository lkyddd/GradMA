from lightfed.core.mode import BaseNode
from mpi4py import MPI


# 锚定req对象，否则python会立刻回收req对象，导致传输失败
# 还有一种异步发送方法：
#   MPI.Attach_buffer(np.empty(BUFFER_SIZE, dtype=np.int))
#   comm.ibsend(obj, dest)
# 但是这样需要设置BUFFER_SIZE，大了浪费，小了不够
class _RequestManager:
    def __init__(self):
        self.req_list = []

    def put(self, req):
        self.req_list.append(req)
        self._expire_()

    def _expire_(self):
        self.req_list = [req for req in self.req_list if not req.Get_status()]


_REQ_MANAGER_ = _RequestManager()


class ClusterNode(BaseNode):
    def __init__(self, rank):
        super().__init__(rank)
        self._comm_ = MPI.COMM_WORLD

    def __getattr__(self, func_name):
        return lambda *args, **kwargs: _REQ_MANAGER_.put(self._comm_.isend((func_name, args, kwargs), dest=self._rank_))
