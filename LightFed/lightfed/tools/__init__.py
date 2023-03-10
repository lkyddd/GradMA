class PeriodicCounter:
    def __init__(self, period):
        self.period = period
        # self.count = 1

    def is_integrity_period(self, step):
        return not step % self.period

    # def add(self, increment=1):
    #     self.count += increment

    # def clear(self):
    #     self.count = 1
