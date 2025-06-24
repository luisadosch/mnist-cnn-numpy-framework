
from typing import List
from tensor import Tensor

class LossInterface():

    def __init__(self):
        pass

    def forward(self, predictions: List[Tensor], labels: List[Tensor]) -> float:
        pass

    def backward(self, predictions: List[Tensor], labels: List[Tensor]):
        pass