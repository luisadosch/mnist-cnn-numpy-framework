from tensor import Tensor
import json
from typing import List


class LayerInterface:
    def __init__(self):
        pass

    def forward(self, in_tensors: List[Tensor], out_tensors: List[Tensor]):
        pass

    def backward(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass

    def calculate_delta_weights(self, out_tensors: List[Tensor], in_tensors: List[Tensor]):
        pass

    def save_params(self, path):
        pass

    def load_params(self, path):
        pass

    def to_dict(self):
        return {
            "type": str(type(self)),
        }

    @classmethod
    def from_dict(cls, data):
        layer = cls()
        return layer
