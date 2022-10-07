from typing import Union, Mapping

from torch import Tensor
from torchmetrics import Metric

Number = Union[int, float]
Metric = Union[Metric, Tensor, Number]
MetricCollection = Union[Metric, Mapping[str, Metric]]
