import logging
from lightfed.core.mode import BaseRunner
from .local_comm import MESSAGE_QUEUE


class LocalRunner(BaseRunner):
    def __init__(self, server_list):
        self.server_list = server_list

    def _start_servers_(self):
        for server in self.server_list:
            server.start()

    def _any_survival_(self):
        return any(self.server_list)

    def _is_survival_(self, rank):
        return self.server_list[rank] is not None

    def _kill_server(self, rank):
        if self._is_survival_(rank):
            self.server_list[rank].end()
            self.server_list[rank] = None

    def _kill_if_finished_(self, rank):
        if self._is_survival_(rank) and self.server_list[rank].end_condition():
            self._kill_server(rank)

    def _kill_finished_servers(self):
        for rank in range(len(self.server_list)):
            self._kill_if_finished_(rank)

    def _kill_all_servers(self):
        for rank in range(len(self.server_list)):
            self._kill_server(rank)

    def run(self):
        self._start_servers_()
        self._kill_finished_servers()
        while self._any_survival_():
            if len(MESSAGE_QUEUE) == 0:
                logging.warning(f"MESSAGE QUEUE is empty but there are some survival servers")
                self._kill_all_servers()
                break

            target_rank, func_name, args, kwargs = MESSAGE_QUEUE.pop(0)
            if func_name == "__shutdown__":
                self._kill_server(target_rank)
            elif self._is_survival_(target_rank):
                getattr(self.server_list[target_rank], func_name)(*args, **kwargs)
                self._kill_if_finished_(target_rank)

        MESSAGE_QUEUE.clear()
