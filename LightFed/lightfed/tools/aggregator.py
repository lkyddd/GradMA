import copy
from abc import ABC, abstractmethod


class BaseAgg(ABC):
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def merge(self, other):
        pass

    @abstractmethod
    def put(self, *args, **kwargs):
        pass

    @abstractmethod
    def clear(self):
        pass

    def get_and_clear(self):
        _res = self.get()
        self.clear()
        return _res

    def __repr__(self):
        return str(self.get())

    def __str__(self):
        return self.__repr__()


class ModelStateAvgAgg(BaseAgg):
    def __init__(self):
        self.accum_model_state = None
        self.accum_weight = 0.0

    def put(self, state, weight=1.0):  # 这里的weight表示sample_number
        if self.accum_model_state is None:
            self.accum_model_state = copy.deepcopy(state)
            if weight != 1.0:
                for k in self.accum_model_state.keys():
                    self.accum_model_state[k].data *= weight
        else:
            for k in self.accum_model_state.keys():
                self.accum_model_state[k].data += state[k].data * weight
        self.accum_weight += weight
        return self

    def merge(self, other):
        if self.accum_model_state is None:
            self.accum_model_state = copy.deepcopy(other.accum_model_state)
            self.accum_weight = other.accum_weight
        else:
            for k in self.accum_model_state.keys():
                self.accum_model_state[k] += other.accum_model_state[k]
            self.accum_weight += other.accum_weight
        return self

    def get(self):
        if self.accum_weight == 0.0:
            return None
        avg_state = copy.deepcopy(self.accum_model_state)
        for k in self.accum_model_state.keys():
            _data = avg_state[k]
            avg_state[k] = (_data / self.accum_weight).to(_data.dtype)
        return avg_state

    def clear(self):
        self.accum_model_state = None
        self.accum_weight = 0.0


class ModelGradAvgAgg(BaseAgg):
    def __init__(self):
        self.accum_model_grad = None
        self.accum_weight = 0.0

    def put(self, state, weight=1.0):  # 这里的weight表示sample_number
        if self.accum_model_grad is None:
            self.accum_model_grad = copy.deepcopy(state)
            if weight != 1.0:
                for k in self.accum_model_grad.keys():
                    self.accum_model_grad[k].data *= weight
        else:
            for k in self.accum_model_grad.keys():
                self.accum_model_grad[k].data += state[k].data * weight
        self.accum_weight += weight
        return self

    def merge(self, other):
        if self.accum_model_grad is None:
            self.accum_model_grad = copy.deepcopy(other.accum_model_grad)
            self.accum_weight = other.accum_weight
        else:
            for k in self.accum_model_grad.keys():
                self.accum_model_grad[k] += other.accum_model_grad[k]
            self.accum_weight += other.accum_weight
        return self

    def get(self):
        if self.accum_weight == 0.0:
            return None
        avg_grad = copy.deepcopy(self.accum_model_grad)
        for k in avg_grad.keys():
            avg_grad[k].data /= self.accum_weight
        return avg_grad

    def clear(self):
        self.accum_model_grad = None
        self.accum_weight = 0.0


class AggregatorTable(BaseAgg):
    def __init__(self):
        self.agg_dict = {}

    def put(self, key, aggregator):
        if key in self.agg_dict:
            self.agg_dict[key].merge(aggregator)
        else:
            self.agg_dict[key] = aggregator
        return self

    def merge(self, other):
        for key, agg in other.agg_dict.items():
            self.put(key, agg)
        return self

    def get(self):
        return {key: agg.get() for key, agg in self.agg_dict.items()}

    def clear(self):
        self.agg_dict = {}


class NumericAvgAgg(BaseAgg):
    def __init__(self):
        self.accum_numeric = 0.0
        self.accum_weight = 0.0

    def put(self, numeric, weight=1.0):
        self.accum_numeric += numeric * weight
        self.accum_weight += weight
        return self

    def merge(self, other):
        self.accum_numeric += other.accum_numeric
        self.accum_weight += other.accum_weight
        return self

    def get(self):
        if self.accum_weight == 0.0:
            return None
        return self.accum_numeric / self.accum_weight

    def clear(self):
        self.accum_numeric = 0.0
        self.accum_weight = 0.0


class NumericVarAgg(BaseAgg):
    def __init__(self):
        self.accum_numeric = 0.0
        self.accum_square_numeric = 0.0
        self.accum_weight = 0.0

    def put(self, numeric, weight=1.0):
        self.accum_numeric += numeric * weight
        self.accum_square_numeric += numeric * numeric * weight
        self.accum_weight += weight
        return self

    def merge(self, other):
        self.accum_numeric += other.accum_numeric
        self.accum_square_numeric += other.accum_square_numeric
        self.accum_weight += other.accum_weight
        return self

    def get(self):
        if self.accum_weight == 0.0:
            return None
        return (self.accum_square_numeric / self.accum_weight) - ((self.accum_numeric / self.accum_weight) ** 2)

    def clear(self):
        self.accum_numeric = 0.0
        self.accum_square_numeric = 0.0
        self.accum_weight = 0.0


class NumericMaxAgg(BaseAgg):
    def __init__(self):
        self.max_value = None

    def put(self, numeric):
        if self.max_value is None:
            self.max_value = numeric
        else:
            self.max_value = max(self.max_value, numeric)
        return self

    def merge(self, other):
        self.put(other.max_value)
        return self

    def get(self):
        return self.max_value

    def clear(self):
        self.max_value = None


class NumericMinAgg(BaseAgg):
    def __init__(self):
        self.min_value = None

    def put(self, numeric):
        if self.min_value is None:
            self.min_value = numeric
        else:
            self.min_value = min(self.min_value, numeric)
        return self

    def get(self):
        return self.min_value

    def merge(self, other):
        self.put(other.min_value)
        return self

    def clear(self):
        self.min_value = None


class NumericSumAgg(BaseAgg):
    def __init__(self):
        self.value = 0.0

    def put(self, numeric):
        self.value += numeric
        return self

    def merge(self, other):
        self.put(other.value)
        return self

    def get(self):
        return self.value

    def clear(self):
        self.value = 0.0


class UDFAgg(BaseAgg):
    def __init__(self, agg_func=lambda item_list: item_list):
        self.item_list = []
        self.agg_func = agg_func

    def put(self, item):
        self.item_list.append(item)
        return self

    def merge(self, other):
        self.item_list.extend(other.item_list)
        return self

    def clear(self):
        self.item_list.clear()
        return self

    def get(self):
        return self.agg_func(self.item_list) if self.item_list else None
