from lightfed.core.mode import BaseRunner
from mpi4py import MPI


class ClusterRunner(BaseRunner):
    def __init__(self, server):
        self.server = server
        self.comm = MPI.COMM_WORLD

    def run(self):
        self.server.start()
        while not self.server.end_condition():
            func_name, args, kwargs = self.comm.recv()
            if func_name == "__shutdown__":
                break
            getattr(self.server, func_name)(*args, **kwargs)
        self.server.end()
        self.comm.Barrier()
