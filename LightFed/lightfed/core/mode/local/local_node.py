import copy
from lightfed.core.mode import BaseNode
from .local_comm import MESSAGE_QUEUE


class LocalNode(BaseNode):
    def __init__(self, rank):
        super().__init__(rank)

    def __getattr__(self, func_name):

        def _send_func(*args, **kwargs):
            if self._deepcopy:
                args = copy.deepcopy(args)
                kwargs = copy.deepcopy(kwargs)
            MESSAGE_QUEUE.append((self._rank_, func_name, args, kwargs))
            self._reset()

        return _send_func
